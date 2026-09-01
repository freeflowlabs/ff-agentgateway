use agent_core::strng;
use agent_core::strng::Strng;
use serde_json::{Map, Value};

use crate::llm::{AIError, RouteType};
use crate::*;

const ANTHROPIC_VERSION: &str = "vertex-2023-10-16";

/// The Vertex `us` multi-region, used as the default when no region is configured.
///
/// Upstream defaults an unset region to `global`. We do not: model inference runs on
/// customer content and must be served from the United States, and `global` is not a
/// region — Vertex may serve a global-endpoint request from wherever the model is
/// available, so it carries no US-processing commitment.
const US_MULTI_REGION: &str = "us";

/// True when `region` names the US multi-region or a US region.
///
/// Any `us-<region>` (us-central1, us-east4, us-west1, ...) is United States; matching the
/// prefix rather than enumerating regions means a newly opened US region needs no code
/// change. Other locales do not use this prefix — Canada is `northamerica-*` — so they are
/// correctly excluded.
fn is_us_region(region: &str) -> bool {
	let region = region.trim().to_ascii_lowercase();
	region == US_MULTI_REGION || region.starts_with("us-")
}

/// Reject a non-US Vertex region at config-parse time, so the gateway refuses to start
/// rather than proxying customer content out of the United States.
///
/// This is deliberately fatal instead of a warning-and-override: a silently corrected
/// value hides the misconfiguration, and there is no case where serving elsewhere is the
/// desired outcome. There is no override flag.
fn de_us_region<'de, D>(deserializer: D) -> Result<Option<Strng>, D::Error>
where
	D: serde::Deserializer<'de>,
{
	use serde::Deserialize;
	let region = Option::<Strng>::deserialize(deserializer)?;
	match &region {
		// Absent is allowed and resolves to the US multi-region (see get_host).
		None => Ok(region),
		// Normalised on the way in: the value is interpolated into the upstream host and
		// into the locations/<region> path, and Vertex expects it lower-case.
		Some(r) if is_us_region(r) => Ok(Some(strng::new(r.trim().to_ascii_lowercase()))),
		Some(r) => Err(serde::de::Error::custom(format!(
			"vertex region {r:?} is not a US location; refusing to start. Model inference \
			 must be served from the US: set the region to {US_MULTI_REGION:?} or a 'us-' \
			 region. Note that 'global' is not a region — Vertex may serve it from any \
			 region where the model is available, so it carries no US-processing commitment."
		))),
	}
}

#[apply(schema!)]
pub struct Provider {
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub model: Option<Strng>,
	/// Vertex serving location. Validated on parse: a non-US value fails config load and
	/// the gateway does not start. Unset resolves to the US multi-region, never `global`.
	#[serde(
		default,
		skip_serializing_if = "Option::is_none",
		deserialize_with = "de_us_region"
	)]
	pub region: Option<Strng>,
	pub project_id: Strng,
}

impl super::Provider for Provider {
	const NAME: Strng = strng::literal!("gcp.vertex_ai");
}

impl Provider {
	fn configured_model<'a>(&'a self, request_model: Option<&'a str>) -> Option<&'a str> {
		self.model.as_deref().or(request_model)
	}

	pub fn is_anthropic_model(&self, request_model: Option<&str>) -> bool {
		self.anthropic_model(request_model).is_some()
	}

	pub fn prepare_anthropic_message_body(&self, body: Vec<u8>) -> Result<Vec<u8>, AIError> {
		self.prepare_anthropic_body(body, |b| {
			b.remove("model");
		})
	}

	pub fn prepare_anthropic_count_tokens_body(&self, body: Vec<u8>) -> Result<Vec<u8>, AIError> {
		self.prepare_anthropic_body(body, |b| {
			if let Some(Value::String(model)) = b.get("model") {
				let normalized = self
					.configured_model(Some(model))
					.map(|s| s.to_string())
					.unwrap_or_else(|| model.clone());
				b.insert("model".to_string(), Value::String(normalized));
			}
		})
	}

	/// Shared pipeline for Vertex Anthropic requests: parse, inject version,
	/// apply caller-specific model handling, strip unsupported fields, serialize.
	fn prepare_anthropic_body(
		&self,
		body: Vec<u8>,
		apply: impl FnOnce(&mut Map<String, Value>),
	) -> Result<Vec<u8>, AIError> {
		let mut body: Map<String, Value> =
			serde_json::from_slice(&body).map_err(AIError::RequestMarshal)?;
		body.insert(
			"anthropic_version".to_string(),
			Value::String(ANTHROPIC_VERSION.to_string()),
		);
		apply(&mut body);
		remove_unsupported_vertex_fields(&mut body);
		serde_json::to_vec(&body).map_err(AIError::RequestMarshal)
	}

	pub fn get_path_for_model(
		&self,
		route: RouteType,
		request_model: Option<&str>,
		streaming: bool,
	) -> Strng {
		// Unset resolves to the US multi-region, not `global`: an omitted region must not
		// silently opt the deployment out of US serving. A configured region is already
		// known to be a US one — de_us_region rejects anything else at parse time.
		let location = self
			.region
			.clone()
			.unwrap_or_else(|| strng::literal!("us"));

		match (route, self.anthropic_model(request_model)) {
			(RouteType::AnthropicTokenCount, _) => {
				strng::format!(
					"/v1/projects/{}/locations/{}/publishers/anthropic/models/count-tokens:rawPredict",
					self.project_id,
					location
				)
			},
			(RouteType::Embeddings, _) => {
				let model = self.configured_model(request_model).unwrap_or_default();
				strng::format!(
					"/v1/projects/{}/locations/{}/publishers/google/models/{}:predict",
					self.project_id,
					location,
					model
				)
			},
			(_, Some(model)) => {
				strng::format!(
					"/v1/projects/{}/locations/{}/publishers/anthropic/models/{}:{}",
					self.project_id,
					location,
					model,
					if streaming {
						"streamRawPredict"
					} else {
						"rawPredict"
					}
				)
			},
			_ => {
				strng::format!(
					"/v1/projects/{}/locations/{}/endpoints/openapi/chat/completions",
					self.project_id,
					location
				)
			},
		}
	}

	/// Upstream host for the configured region.
	///
	/// There is no longer a bare `aiplatform.googleapis.com` arm: that host is the global
	/// endpoint, and a `global` region cannot be configured (de_us_region rejects it) nor
	/// defaulted to (unset resolves to the US multi-region).
	///
	/// The `us` multi-region has no `us-aiplatform.googleapis.com` host — Vertex rejects
	/// that hostname with 400 INVALID_ARGUMENT (verified live 2026-09-01). Multi-regions
	/// are served on the regional-endpoint form `aiplatform.us.rep.googleapis.com`, which
	/// is also the host the google-genai SDK (>= 2.x) dials for `us`, and which carries
	/// the in-region processing guarantee at the host level. Single regions keep the
	/// `{region}-aiplatform.googleapis.com` form.
	pub fn get_host(&self, _request_model: Option<&str>) -> Strng {
		match self.region.as_deref() {
			None | Some(US_MULTI_REGION) => {
				strng::format!("aiplatform.{US_MULTI_REGION}.rep.googleapis.com")
			},
			Some(region) => strng::format!("{region}-aiplatform.googleapis.com"),
		}
	}

	fn anthropic_model<'a>(&'a self, request_model: Option<&'a str>) -> Option<Strng> {
		let model = self.configured_model(request_model)?;

		// Strip known prefixes
		let model: &str = model
			.split_once("publishers/anthropic/models/")
			.map(|(_, m)| m)
			.or_else(|| model.strip_prefix("anthropic/"))
			.or_else(|| {
				if model.starts_with("claude-") {
					Some(model)
				} else {
					None
				}
			})?;

		// Replace -YYYYMMDD with @YYYYMMDD
		if model.len() > 8 && model.as_bytes()[model.len() - 9] == b'-' {
			let (base, date) = model.split_at(model.len() - 8);
			if date.chars().all(|c| c.is_ascii_digit()) {
				Some(strng::new(format!("{}@{}", &base[..base.len() - 1], date)))
			} else {
				Some(strng::new(model))
			}
		} else {
			Some(strng::new(model))
		}
	}
}

fn remove_unsupported_vertex_fields(body: &mut Map<String, Value>) {
	body.remove("output_config");
	body.remove("output_format");
	// Vertex supports cache_control but not the "scope" child from the prompt-caching-scope beta.
	for value in body.values_mut() {
		remove_nested_field(value, "cache_control", "scope");
	}
}

fn remove_nested_field(value: &mut Value, key: &str, child: &str) {
	match value {
		Value::Object(map) => {
			if let Some(Value::Object(nested)) = map.get_mut(key) {
				nested.remove(child);
			}
			for v in map.values_mut() {
				remove_nested_field(v, key, child);
			}
		},
		Value::Array(arr) => {
			for v in arr {
				remove_nested_field(v, key, child);
			}
		},
		_ => {},
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	#[rstest::rstest]
	#[case::strip_publishers_prefix(
		Some("publishers/anthropic/models/claude-sonnet-4-5-20251001"),
		None,
		Some("claude-sonnet-4-5@20251001")
	)]
	#[case::strip_anthropic_prefix(
		Some("anthropic/claude-haiku-4-5-20251001"),
		None,
		Some("claude-haiku-4-5@20251001")
	)]
	#[case::raw_claude_prefix(None, Some("claude-opus-3-20240229"), Some("claude-opus-3@20240229"))]
	#[case::no_date_suffix(None, Some("claude-opus-4-6"), Some("claude-opus-4-6"))]
	#[case::legacy_model(
		None,
		Some("claude-3-5-sonnet-20241022"),
		Some("claude-3-5-sonnet@20241022")
	)]
	#[case::non_digit_date_suffix(
		None,
		Some("claude-haiku-4-5-2025abcd"),
		Some("claude-haiku-4-5-2025abcd")
	)]
	#[case::non_anthropic_model(None, Some("text-embedding-004"), None)]
	#[case::provider_model_precedence(
		Some("anthropic/claude-haiku-4-5-20251001"),
		Some("anthropic/claude-sonnet-4-5-20251001"),
		Some("claude-haiku-4-5@20251001")
	)]
	fn test_anthropic_model_normalization(
		#[case] provider: Option<&str>,
		#[case] req: Option<&str>,
		#[case] expected: Option<&str>,
	) {
		let p = Provider {
			project_id: strng::new("test-project"),
			model: provider.map(strng::new),
			region: None,
		};
		let actual = p.anthropic_model(req).map(|m| m.to_string());
		assert_eq!(actual.as_deref(), expected);
	}

	#[rstest::rstest]
	// Unset resolves to the US multi-region, not the global endpoint. The multi-region is
	// served on the .rep. regional-endpoint host — `us-aiplatform.googleapis.com` does not
	// exist (Vertex rejects it with 400 INVALID_ARGUMENT).
	#[case::no_region(None, "aiplatform.us.rep.googleapis.com")]
	#[case::us_multi_region(Some("us"), "aiplatform.us.rep.googleapis.com")]
	#[case::regional(Some("us-central1"), "us-central1-aiplatform.googleapis.com")]
	fn test_get_host(#[case] region: Option<&str>, #[case] expected: &str) {
		let p = Provider {
			project_id: strng::new("test-project"),
			model: None,
			region: region.map(strng::new),
		};
		assert_eq!(p.get_host(None).as_str(), expected);
	}

	#[rstest::rstest]
	#[case::us_multi_region("us")]
	#[case::us_central("us-central1")]
	#[case::us_east("us-east4")]
	fn test_us_region_accepted(#[case] region: &str) {
		let json = serde_json::json!({"projectId": "test-project", "region": region});
		let p: Provider = serde_json::from_value(json).expect("US region must parse");
		assert_eq!(p.region.as_deref(), Some(region));
	}

	#[test]
	fn test_region_is_normalised_to_lower_case() {
		// The value is interpolated into the host and the locations/<region> path, so an
		// upper-case config value must not produce locations/US.
		let json = serde_json::json!({"projectId": "test-project", "region": "US-Central1"});
		let p: Provider = serde_json::from_value(json).expect("US region must parse");
		assert_eq!(p.region.as_deref(), Some("us-central1"));
		assert_eq!(p.get_host(None).as_str(), "us-central1-aiplatform.googleapis.com");
	}

	#[rstest::rstest]
	// `global` is refused by name: it is not a region and carries no US-processing
	// commitment, so it must not be configurable.
	#[case::global("global")]
	#[case::europe("europe-west1")]
	#[case::canada("northamerica-northeast1")]
	#[case::asia("asia-east1")]
	fn test_non_us_region_refused(#[case] region: &str) {
		let json = serde_json::json!({"projectId": "test-project", "region": region});
		let err = serde_json::from_value::<Provider>(json)
			.expect_err("a non-US region must fail config parse so the gateway does not start");
		assert!(
			err.to_string().contains("not a US location"),
			"unexpected error: {err}"
		);
	}

	#[test]
	fn test_absent_region_parses_and_defaults_to_us() {
		let json = serde_json::json!({"projectId": "test-project"});
		let p: Provider = serde_json::from_value(json).expect("absent region is allowed");
		assert_eq!(p.region, None);
		assert_eq!(p.get_host(None).as_str(), "aiplatform.us.rep.googleapis.com");
	}

	#[test]
	fn test_remove_top_level_output_fields() {
		let mut body: Map<String, Value> = serde_json::from_value(serde_json::json!({
			"model": "claude-sonnet-4-5-20251001",
			"output_config": {"format": "json"},
			"output_format": "markdown",
			"messages": [{"role": "user", "content": "hello"}]
		}))
		.unwrap();
		remove_unsupported_vertex_fields(&mut body);
		assert!(!body.contains_key("output_config"));
		assert!(!body.contains_key("output_format"));
		assert!(body.contains_key("model"));
		assert!(body.contains_key("messages"));
	}

	#[test]
	fn test_output_fields_preserved_when_nested() {
		let mut body: Map<String, Value> = serde_json::from_value(serde_json::json!({
			"messages": [{
				"role": "user",
				"content": "hello",
				"output_config": {"format": "json"},
				"output_format": "markdown"
			}]
		}))
		.unwrap();
		remove_unsupported_vertex_fields(&mut body);
		let msg = body["messages"][0].as_object().unwrap();
		assert!(msg.contains_key("output_config"));
		assert!(msg.contains_key("output_format"));
	}

	#[test]
	fn test_cache_control_scope_removed_recursively() {
		let mut body: Map<String, Value> = serde_json::from_value(serde_json::json!({
			"system": [{
				"type": "text",
				"text": "You are helpful.",
				"cache_control": {"type": "ephemeral", "scope": "turn"}
			}],
			"messages": [{
				"role": "user",
				"content": [{
					"type": "text",
					"text": "hello",
					"cache_control": {"type": "ephemeral", "scope": "session"}
				}]
			}]
		}))
		.unwrap();
		remove_unsupported_vertex_fields(&mut body);
		let sys_cc = body["system"][0]["cache_control"].as_object().unwrap();
		assert_eq!(sys_cc.get("type").unwrap(), "ephemeral");
		assert!(!sys_cc.contains_key("scope"));
		let msg_cc = body["messages"][0]["content"][0]["cache_control"]
			.as_object()
			.unwrap();
		assert_eq!(msg_cc.get("type").unwrap(), "ephemeral");
		assert!(!msg_cc.contains_key("scope"));
	}

	#[test]
	fn test_cache_control_without_scope_untouched() {
		let mut body: Map<String, Value> = serde_json::from_value(serde_json::json!({
			"messages": [{
				"role": "user",
				"content": [{
					"type": "text",
					"text": "hello",
					"cache_control": {"type": "ephemeral"}
				}]
			}]
		}))
		.unwrap();
		let expected = body.clone();
		remove_unsupported_vertex_fields(&mut body);
		assert_eq!(body, expected);
	}

	#[test]
	fn test_cache_control_non_object_untouched() {
		let mut body: Map<String, Value> = serde_json::from_value(serde_json::json!({
			"messages": [{
				"role": "user",
				"content": [{
					"type": "text",
					"text": "hello",
					"cache_control": "enabled"
				}]
			}]
		}))
		.unwrap();
		let expected = body.clone();
		remove_unsupported_vertex_fields(&mut body);
		assert_eq!(body, expected);
	}

	#[test]
	fn test_realistic_anthropic_messages_body() {
		let mut body: Map<String, Value> = serde_json::from_value(serde_json::json!({
			"model": "claude-sonnet-4-5-20251001",
			"max_tokens": 1024,
			"output_config": {"format": "json"},
			"output_format": "markdown",
			"system": [{
				"type": "text",
				"text": "You are a helpful assistant.",
				"cache_control": {"type": "ephemeral", "scope": "turn"}
			}],
			"messages": [
				{
					"role": "user",
					"content": [
						{
							"type": "text",
							"text": "What is 2+2?",
							"cache_control": {"type": "ephemeral", "scope": "session"}
						},
						{
							"type": "image",
							"source": {"type": "base64", "data": "abc"},
							"cache_control": {"type": "ephemeral"}
						}
					]
				},
				{
					"role": "assistant",
					"content": [{"type": "text", "text": "4"}]
				}
			]
		}))
		.unwrap();
		remove_unsupported_vertex_fields(&mut body);

		// Top-level fields removed
		assert!(!body.contains_key("output_config"));
		assert!(!body.contains_key("output_format"));
		// Preserved fields
		assert_eq!(body["max_tokens"], 1024);
		assert_eq!(body["model"], "claude-sonnet-4-5-20251001");

		// System cache_control: scope removed, type kept
		let sys_cc = body["system"][0]["cache_control"].as_object().unwrap();
		assert_eq!(sys_cc.len(), 1);
		assert_eq!(sys_cc["type"], "ephemeral");

		// First user content block: scope removed
		let user_cc = body["messages"][0]["content"][0]["cache_control"]
			.as_object()
			.unwrap();
		assert_eq!(user_cc.len(), 1);
		assert_eq!(user_cc["type"], "ephemeral");

		// Second user content block: no scope, so unchanged (still has type)
		let img_cc = body["messages"][0]["content"][1]["cache_control"]
			.as_object()
			.unwrap();
		assert_eq!(img_cc.len(), 1);
		assert_eq!(img_cc["type"], "ephemeral");

		// Assistant content untouched (no cache_control)
		assert!(
			body["messages"][1]["content"][0]
				.get("cache_control")
				.is_none()
		);
	}
}
