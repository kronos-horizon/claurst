// Team tools: create and disband multi-agent swarm teams.
//
// TeamCreateTool — create a named team, run N AgentTool sub-agents in parallel
//                  via the globally-injected AgentRunner, and return aggregated
//                  results from every agent.
// TeamDeleteTool — cancel / clean up a named team.
//
// Architecture note
// -----------------
// cc-tools cannot depend on cc-query (that would be circular: cc-query already
// depends on cc-tools).  We therefore use a dependency-injection pattern:
//
//   1. cc-tools exposes `register_agent_runner(f)` which stores a callable in a
//      process-global slot.
//   2. cc-query calls `register_agent_runner` at process startup, passing a
//      closure that invokes `run_query_loop`.
//   3. TeamCreateTool calls `run_agent(...)` which dispatches through that slot.
//
// This keeps the module self-contained and avoids any extra crate boundary.

use crate::{PermissionLevel, Tool, ToolContext, ToolResult};
use async_trait::async_trait;
use futures::future::join_all;
use once_cell::sync::OnceCell;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use uuid::Uuid;

/// Maximum number of nested `TeamCreate` invocations allowed before refusing.
pub const MAX_TEAM_DEPTH: u32 = 3;

/// Per-agent wall-clock deadline.  Production: 2 minutes.  Tests: 2 seconds.
#[cfg(not(test))]
const AGENT_TIMEOUT_SECS: u64 = 120;
#[cfg(test)]
const AGENT_TIMEOUT_SECS: u64 = 2;

// ---------------------------------------------------------------------------
// Global agent-runner injection
// ---------------------------------------------------------------------------

/// A boxed async function that runs an agent sub-task and returns its output.
///
/// Arguments:
///   description — short label for logging
///   prompt      — full task prompt
///   tools       — optional allowlist of tool names; None means all tools
///   system      — optional system prompt override
///   max_turns   — max agent turns (default 10 when None)
///   ctx         — parent tool context (cloned in for the sub-agent)
///
/// Returns the agent's final text output.
pub type AgentRunFn = Arc<
    dyn Fn(
            String,                // description
            String,                // prompt
            Option<Vec<String>>,   // tools allowlist
            Option<String>,        // system prompt
            Option<u32>,           // max_turns
            Arc<ToolContext>,      // context
        ) -> Pin<Box<dyn Future<Output = String> + Send>>
        + Send
        + Sync,
>;

static AGENT_RUNNER: OnceCell<AgentRunFn> = OnceCell::new();

/// Test-only override: when set, `run_agent` bypasses the global AGENT_RUNNER.
/// Protected by TEST_LOCK (tokio::sync::Mutex) so tests that set it don't
/// race with each other across async await points.
#[cfg(test)]
static TEST_AGENT_RUNNER: once_cell::sync::Lazy<parking_lot::Mutex<Option<AgentRunFn>>> =
    once_cell::sync::Lazy::new(|| parking_lot::Mutex::new(None));

/// Tokio mutex used by tests to serialise access to TEST_AGENT_RUNNER.
/// Held across `.await` points (tokio::sync::MutexGuard is Send).
#[cfg(test)]
static TEST_LOCK: once_cell::sync::Lazy<tokio::sync::Mutex<()>> =
    once_cell::sync::Lazy::new(|| tokio::sync::Mutex::new(()));

/// Register the global agent runner.  Called once at process startup by cc-query.
///
/// # Panics
/// Panics if called more than once (once_cell semantics).
pub fn register_agent_runner(f: AgentRunFn) {
    if AGENT_RUNNER.set(f).is_err() {
        panic!("register_agent_runner called more than once");
    }
}

/// Execute a sub-agent via the registered runner.
///
/// Falls back to a stub result when no runner has been registered (e.g., in
/// unit tests that don't initialise cc-query).
async fn run_agent(
    description: String,
    prompt: String,
    tools: Option<Vec<String>>,
    system: Option<String>,
    max_turns: Option<u32>,
    ctx: Arc<ToolContext>,
) -> String {
    #[cfg(test)]
    {
        // Clone the Arc out of the lock so we don't hold it across the await.
        let runner_opt: Option<AgentRunFn> = TEST_AGENT_RUNNER.lock().clone();
        if let Some(runner) = runner_opt {
            return runner(description, prompt, tools, system, max_turns, ctx).await;
        }
    }
    if let Some(runner) = AGENT_RUNNER.get() {
        runner(description, prompt, tools, system, max_turns, ctx).await
    } else {
        "[No agent runner registered — cc-query not initialised]".to_string()
    }
}

// ---------------------------------------------------------------------------
// Active-team registry
// ---------------------------------------------------------------------------
//
// Maps sanitized_team_name -> list of per-agent cancel tokens so that
// TeamDeleteTool can signal cancellation to still-running agents.

use dashmap::DashMap;
use once_cell::sync::Lazy;
use tokio_util::sync::CancellationToken;

static ACTIVE_TEAMS: Lazy<DashMap<String, Vec<CancellationToken>>> =
    Lazy::new(DashMap::new);

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

fn teams_base_dir() -> Option<std::path::PathBuf> {
    dirs::home_dir().map(|h| h.join(".claurst").join("teams"))
}

fn team_dir(team_name: &str) -> Option<std::path::PathBuf> {
    teams_base_dir().map(|b| b.join(sanitize_name(team_name)))
}

/// Sanitize a team name to a safe directory component.
fn sanitize_name(name: &str) -> String {
    name.chars()
        .map(|c| {
            if c.is_alphanumeric() || c == '-' || c == '_' {
                c
            } else {
                '-'
            }
        })
        .collect()
}

fn now_millis() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

// ---------------------------------------------------------------------------
// On-disk schema
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
struct TeamMember {
    agent_id: String,
    name: String,
    role: String,
    joined_at: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<String>>,
}

#[derive(Debug, Serialize, Deserialize)]
struct TeamConfig {
    name: String,
    task: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    description: Option<String>,
    created_at: u64,
    lead_agent_id: String,
    lead_session_id: String,
    parallel: bool,
    members: Vec<TeamMember>,
}

// ---------------------------------------------------------------------------
// TeamCreateTool
// ---------------------------------------------------------------------------

pub struct TeamCreateTool;

/// Per-agent specification provided in the input.
#[derive(Debug, Deserialize)]
struct AgentSpec {
    name: String,
    #[serde(default)]
    role: Option<String>,
    #[serde(default)]
    tools: Option<Vec<String>>,
    /// Optional per-agent task override.  When absent the shared top-level
    /// `task` is used.
    #[serde(default)]
    task: Option<String>,
}

#[derive(Debug, Deserialize)]
struct TeamCreateInput {
    team_name: String,
    /// The shared task all agents work on (individual agents may override via
    /// `agents[i].task`).
    task: String,
    /// List of agents to spawn.
    #[serde(default)]
    agents: Vec<AgentSpec>,
    /// When true (default) all agents run in parallel via join_all.
    /// When false they run sequentially.
    #[serde(default = "default_parallel")]
    parallel: bool,
    /// Optional description stored in the config file.
    #[serde(default)]
    description: Option<String>,
}

fn default_parallel() -> bool {
    true
}

#[async_trait]
impl Tool for TeamCreateTool {
    fn name(&self) -> &str {
        "TeamCreate"
    }

    fn description(&self) -> &str {
        "Create a named team of agents that collectively work on a shared task. \
         Each agent gets a restricted tool list and its own prompt. \
         Agents run in parallel by default and their outputs are aggregated. \
         Input: { team_name, task, agents: [{name, role?, tools?, task?}], parallel?, description? }"
    }

    fn permission_level(&self) -> PermissionLevel {
        PermissionLevel::Write
    }

    fn input_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "team_name": {
                    "type": "string",
                    "description": "Name for the new team."
                },
                "task": {
                    "type": "string",
                    "description": "The shared task all agents should work on."
                },
                "agents": {
                    "type": "array",
                    "description": "Agent specifications.  Each agent runs independently.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": { "type": "string" },
                            "role": { "type": "string", "description": "Role/persona description." },
                            "tools": {
                                "type": "array",
                                "items": { "type": "string" },
                                "description": "Allowed tool names.  Omit to use all tools."
                            },
                            "task": {
                                "type": "string",
                                "description": "Per-agent task override.  Falls back to top-level task."
                            }
                        },
                        "required": ["name"]
                    }
                },
                "parallel": {
                    "type": "boolean",
                    "description": "Run all agents in parallel (default: true).  Set false for sequential."
                },
                "description": {
                    "type": "string",
                    "description": "Optional team description stored in config."
                }
            },
            "required": ["team_name", "task"]
        })
    }

    async fn execute(&self, input: Value, ctx: &ToolContext) -> ToolResult {
        let params: TeamCreateInput = match serde_json::from_value(input) {
            Ok(p) => p,
            Err(e) => return ToolResult::error(format!("Invalid input: {}", e)),
        };

        if params.team_name.trim().is_empty() {
            return ToolResult::error("team_name is required for TeamCreate".to_string());
        }
        if params.task.trim().is_empty() {
            return ToolResult::error("task is required for TeamCreate".to_string());
        }

        // Enforce recursion depth limit so nested TeamCreate calls cannot cascade.
        if ctx.team_depth >= MAX_TEAM_DEPTH {
            return ToolResult::error(format!(
                "TeamCreate recursion depth limit ({}) exceeded. \
                 Nested team creation beyond depth {} is not permitted.",
                MAX_TEAM_DEPTH, MAX_TEAM_DEPTH
            ));
        }

        let safe_name = sanitize_name(&params.team_name);
        let lead_agent_id = format!("team-lead@{}", safe_name);

        // Resolve team directory, disambiguating if name already exists.
        let dir = match team_dir(&params.team_name) {
            Some(d) => d,
            None => return ToolResult::error("Could not determine home directory".to_string()),
        };

        let (final_name, final_dir) = if dir.exists() {
            let suffix = &Uuid::new_v4().to_string()[..6];
            let new_name = format!("{}-{}", safe_name, suffix);
            let new_dir = match team_dir(&new_name) {
                Some(d) => d,
                None => return ToolResult::error("Could not determine home directory".to_string()),
            };
            (new_name, new_dir)
        } else {
            (safe_name.clone(), dir)
        };

        if let Err(e) = tokio::fs::create_dir_all(&final_dir).await {
            return ToolResult::error(format!("Failed to create team directory: {}", e));
        }

        let now = now_millis();

        // Build the member list for the config file.
        let members: Vec<TeamMember> = params
            .agents
            .iter()
            .enumerate()
            .map(|(i, spec)| TeamMember {
                agent_id: format!("agent-{}@{}", i, final_name),
                name: spec.name.clone(),
                role: spec.role.clone().unwrap_or_else(|| "assistant".to_string()),
                joined_at: now,
                tools: spec.tools.clone(),
            })
            .collect();

        let config = TeamConfig {
            name: final_name.clone(),
            task: params.task.clone(),
            description: params.description.clone(),
            created_at: now,
            lead_agent_id: lead_agent_id.clone(),
            lead_session_id: ctx.session_id.clone(),
            parallel: params.parallel,
            members: members.clone(),
        };

        let config_json = match serde_json::to_string_pretty(&config) {
            Ok(j) => j,
            Err(e) => return ToolResult::error(format!("Serialisation error: {}", e)),
        };

        let config_path = final_dir.join("config.json");
        if let Err(e) = tokio::fs::write(&config_path, &config_json).await {
            return ToolResult::error(format!("Failed to write config.json: {}", e));
        }

        // Write empty results placeholder.
        let results_path = final_dir.join("results.json");
        if let Err(e) = tokio::fs::write(&results_path, "[]").await {
            return ToolResult::error(format!("Failed to write results.json: {}", e));
        }

        // -----------------------------------------------------------------------
        // Spawn agents
        // -----------------------------------------------------------------------
        //
        // If there are no agent specs, return early with just the config info.
        if params.agents.is_empty() {
            let team_file_path = config_path.to_string_lossy().to_string();
            return ToolResult::success(
                json!({
                    "team_name": final_name,
                    "team_file_path": team_file_path,
                    "lead_agent_id": lead_agent_id,
                    "agents_spawned": 0,
                    "results": []
                })
                .to_string(),
            );
        }

        // Create one CancellationToken per agent so TeamDeleteTool can signal stop.
        let cancel_tokens: Vec<CancellationToken> = params
            .agents
            .iter()
            .map(|_| CancellationToken::new())
            .collect();

        ACTIVE_TEAMS.insert(final_name.clone(), cancel_tokens.clone());

        // Increment depth so sub-agents can enforce the recursion limit.
        let mut sub_ctx = ctx.clone();
        sub_ctx.team_depth = ctx.team_depth + 1;
        let ctx_arc = Arc::new(sub_ctx);

        // Build per-agent futures.
        let agent_futures: Vec<_> = params
            .agents
            .iter()
            .enumerate()
            .map(|(i, spec)| {
                let agent_name = spec.name.clone();
                let role = spec.role.clone().unwrap_or_else(|| "assistant".to_string());
                let tools = spec.tools.clone();
                let agent_task = spec
                    .task
                    .clone()
                    .unwrap_or_else(|| params.task.clone());
                let team_name_inner = final_name.clone();
                let cancel = cancel_tokens[i].clone();
                let ctx_inner = ctx_arc.clone();

                let system_prompt = format!(
                    "You are agent '{name}' on team '{team}'.  Your role: {role}.\n\
                     Work on the assigned task thoroughly and return your complete findings.",
                    name = agent_name,
                    team = team_name_inner,
                    role = role,
                );

                let description = format!("{}/{}", team_name_inner, agent_name);

                async move {
                    // Honour cancellation: return early if the team was deleted
                    // before we even start.
                    if cancel.is_cancelled() {
                        return (agent_name, "[Cancelled before start]".to_string());
                    }

                    let timeout_name = agent_name.clone();
                    let result = tokio::select! {
                        out = run_agent(
                            description,
                            agent_task,
                            tools,
                            Some(system_prompt),
                            Some(10),
                            ctx_inner,
                        ) => out,
                        _ = cancel.cancelled() => "[Agent cancelled by TeamDelete]".to_string(),
                        _ = tokio::time::sleep(std::time::Duration::from_secs(AGENT_TIMEOUT_SECS)) => {
                            format!("[Agent '{}' timed out after {}s]", timeout_name, AGENT_TIMEOUT_SECS)
                        }
                    };

                    (agent_name, result)
                }
            })
            .collect();

        // Run agents: parallel (join_all) or sequential (iterate).
        let agent_results: Vec<(String, String)> = if params.parallel {
            join_all(agent_futures).await
        } else {
            let mut results = Vec::with_capacity(agent_futures.len());
            for fut in agent_futures {
                results.push(fut.await);
            }
            results
        };

        // Clean up the active-team registry.
        ACTIVE_TEAMS.remove(&final_name);

        // Persist results to disk.
        let results_json: Vec<Value> = agent_results
            .iter()
            .map(|(name, output)| json!({ "agent": name, "output": output }))
            .collect();
        let _ = tokio::fs::write(
            &results_path,
            serde_json::to_string_pretty(&results_json).unwrap_or_default(),
        )
        .await;

        // Build the aggregated output string.
        let mut aggregated = String::new();
        for (name, output) in &agent_results {
            aggregated.push_str(&format!("## Agent: {}\n\n{}\n\n", name, output));
        }

        let team_file_path = config_path.to_string_lossy().to_string();

        ToolResult::success(
            json!({
                "team_name": final_name,
                "team_file_path": team_file_path,
                "lead_agent_id": lead_agent_id,
                "agents_spawned": agent_results.len(),
                "parallel": params.parallel,
                "results": results_json,
                "aggregated_output": aggregated.trim()
            })
            .to_string(),
        )
    }
}

// ---------------------------------------------------------------------------
// TeamDeleteTool
// ---------------------------------------------------------------------------

pub struct TeamDeleteTool;

#[derive(Debug, Deserialize)]
struct TeamDeleteInput {
    team_name: String,
}

#[async_trait]
impl Tool for TeamDeleteTool {
    fn name(&self) -> &str {
        "TeamDelete"
    }

    fn description(&self) -> &str {
        "Cancel a running team and clean up its directories. \
         Signals all in-flight agents to stop, then removes \
         ~/.claurst/teams/{team_name}/."
    }

    fn permission_level(&self) -> PermissionLevel {
        PermissionLevel::Write
    }

    fn input_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "team_name": {
                    "type": "string",
                    "description": "Name of the team to delete."
                }
            },
            "required": ["team_name"]
        })
    }

    async fn execute(&self, input: Value, _ctx: &ToolContext) -> ToolResult {
        let params: TeamDeleteInput = match serde_json::from_value(input) {
            Ok(p) => p,
            Err(e) => return ToolResult::error(format!("Invalid input: {}", e)),
        };

        if params.team_name.trim().is_empty() {
            return ToolResult::error("team_name is required for TeamDelete".to_string());
        }

        let safe_name = sanitize_name(&params.team_name);

        // Cancel any still-running agents.
        let cancelled_count = if let Some((_, tokens)) = ACTIVE_TEAMS.remove(&safe_name) {
            let count = tokens.len();
            for token in tokens {
                token.cancel();
            }
            count
        } else {
            0
        };

        // Remove the team directory from disk.
        let dir = match team_dir(&params.team_name) {
            Some(d) => d,
            None => return ToolResult::error("Could not determine home directory".to_string()),
        };

        if !dir.exists() {
            // Directory already gone — treat as success if we cancelled agents,
            // or as an informational message if nothing was running.
            return ToolResult::success(
                json!({
                    "success": true,
                    "message": format!(
                        "Team '{}' directory not found (may have been cleaned up already). \
                         Cancelled {} agent(s).",
                        params.team_name, cancelled_count
                    ),
                    "team_name": params.team_name,
                    "cancelled_agents": cancelled_count
                })
                .to_string(),
            );
        }

        if let Err(e) = tokio::fs::remove_dir_all(&dir).await {
            return ToolResult::error(format!(
                "Failed to remove team directory '{}': {}",
                dir.display(),
                e
            ));
        }

        ToolResult::success(
            json!({
                "success": true,
                "message": format!(
                    "Cleaned up team \"{}\" and cancelled {} agent(s).",
                    params.team_name, cancelled_count
                ),
                "team_name": params.team_name,
                "cancelled_agents": cancelled_count
            })
            .to_string(),
        )
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicUsize;

    fn make_ctx(depth: u32) -> ToolContext {
        use claurst_core::config::{Config, PermissionMode};
        use claurst_core::permissions::AutoPermissionHandler;
        ToolContext {
            working_dir: std::path::PathBuf::from("/tmp"),
            permission_mode: PermissionMode::Default,
            permission_handler: Arc::new(AutoPermissionHandler {
                mode: PermissionMode::Default,
            }),
            cost_tracker: claurst_core::cost::CostTracker::new(),
            session_id: "team-test".to_string(),
            file_history: Arc::new(parking_lot::Mutex::new(
                claurst_core::file_history::FileHistory::new(),
            )),
            current_turn: Arc::new(AtomicUsize::new(0)),
            non_interactive: true,
            mcp_manager: None,
            config: Config::default(),
            managed_agent_config: None,
            completion_notifier: None,
            pending_permissions: None,
            permission_manager: None,
            user_question_tx: None,
            team_depth: depth,
        }
    }

    /// Instant stub runner: returns "result:<task>" immediately.
    /// Keyed on task only (not description) so parallel and serial runs over
    /// the same agents produce identical per-agent outputs.
    fn instant_runner() -> AgentRunFn {
        Arc::new(
            |_desc: String,
             task: String,
             _tools: Option<Vec<String>>,
             _sys: Option<String>,
             _max: Option<u32>,
             _ctx: Arc<ToolContext>| {
                Box::pin(async move { format!("result:{}", task) })
                    as Pin<Box<dyn std::future::Future<Output = String> + Send>>
            },
        )
    }

    // -------------------------------------------------------------------------
    // Test 0: Serializability — parallel and serial produce identical per-agent
    // outputs (just potentially in different order).
    // -------------------------------------------------------------------------
    #[tokio::test]
    async fn test_serial_parallel_serializable() {
        let _guard = TEST_LOCK.lock().await;
        *TEST_AGENT_RUNNER.lock() = Some(instant_runner());

        let ctx = make_ctx(0);
        let agents = serde_json::json!([
            {"name": "alpha", "task": "task-A"},
            {"name": "beta",  "task": "task-B"},
            {"name": "gamma", "task": "task-C"},
        ]);

        // Parallel run.
        let r_par = TeamCreateTool
            .execute(
                serde_json::json!({
                    "team_name": "test-ser-par",
                    "task": "shared",
                    "agents": agents,
                    "parallel": true,
                }),
                &ctx,
            )
            .await;
        assert!(!r_par.is_error, "parallel run failed: {}", r_par.content);

        // Serial run.
        let r_ser = TeamCreateTool
            .execute(
                serde_json::json!({
                    "team_name": "test-ser-seq",
                    "task": "shared",
                    "agents": agents,
                    "parallel": false,
                }),
                &ctx,
            )
            .await;
        assert!(!r_ser.is_error, "serial run failed: {}", r_ser.content);

        // Extract and sort per-agent (name, output) pairs from both runs.
        let extract = |content: &str| -> Vec<(String, String)> {
            let v: serde_json::Value = serde_json::from_str(content).unwrap();
            let mut pairs: Vec<(String, String)> = v["results"]
                .as_array()
                .unwrap()
                .iter()
                .map(|r| {
                    (
                        r["agent"].as_str().unwrap().to_string(),
                        r["output"].as_str().unwrap().to_string(),
                    )
                })
                .collect();
            pairs.sort_by(|a, b| a.0.cmp(&b.0));
            pairs
        };

        let par_results = extract(&r_par.content);
        let ser_results = extract(&r_ser.content);

        assert_eq!(
            par_results, ser_results,
            "parallel and serial results differ — serializability violated"
        );

        *TEST_AGENT_RUNNER.lock() = None;
    }

    // -------------------------------------------------------------------------
    // Test 1: Recursion depth — TeamCreate refuses at depth >= MAX_TEAM_DEPTH.
    // -------------------------------------------------------------------------
    #[tokio::test]
    async fn test_recursion_depth_limit() {
        let ctx = make_ctx(MAX_TEAM_DEPTH);
        let result = TeamCreateTool
            .execute(
                serde_json::json!({
                    "team_name": "nested-team",
                    "task": "some task",
                    "agents": [{"name": "bot"}],
                }),
                &ctx,
            )
            .await;
        assert!(
            result.is_error,
            "expected an error when team_depth >= MAX_TEAM_DEPTH"
        );
        assert!(
            result.content.contains("recursion depth limit"),
            "error should mention recursion depth, got: {}",
            result.content
        );
    }

    // -------------------------------------------------------------------------
    // Test 2: Depth is incremented for sub-agents — a depth-2 ctx only has
    // one level of slack left, so a nested call at depth-2 proceeds but the
    // sub-agent's ctx would be depth-3 (blocked on next attempt).
    // -------------------------------------------------------------------------
    #[tokio::test]
    async fn test_depth_incremented_for_sub_agents() {
        let _guard = TEST_LOCK.lock().await;

        // Runner that captures the depth of the ctx it receives.
        let captured_depth = Arc::new(std::sync::Mutex::new(None::<u32>));
        let captured_depth_clone = captured_depth.clone();
        *TEST_AGENT_RUNNER.lock() = Some(Arc::new(
            move |_desc: String,
                  _task: String,
                  _tools: Option<Vec<String>>,
                  _sys: Option<String>,
                  _max: Option<u32>,
                  ctx: Arc<ToolContext>| {
                let captured = captured_depth_clone.clone();
                Box::pin(async move {
                    *captured.lock().unwrap() = Some(ctx.team_depth);
                    "done".to_string()
                }) as Pin<Box<dyn std::future::Future<Output = String> + Send>>
            },
        ));

        let ctx = make_ctx(0);
        let result = TeamCreateTool
            .execute(
                serde_json::json!({
                    "team_name": "test-depth-inc",
                    "task": "t",
                    "agents": [{"name": "a"}],
                }),
                &ctx,
            )
            .await;
        assert!(!result.is_error, "{}", result.content);

        let depth = captured_depth.lock().unwrap().unwrap();
        assert_eq!(
            depth, 1,
            "sub-agent context should have team_depth = parent + 1"
        );

        *TEST_AGENT_RUNNER.lock() = None;
    }

    // -------------------------------------------------------------------------
    // Test 3: Per-agent timeout — agents that exceed AGENT_TIMEOUT_SECS are
    // replaced with a timeout message, not an error return from execute.
    // -------------------------------------------------------------------------
    #[tokio::test]
    async fn test_agent_timeout() {
        let _guard = TEST_LOCK.lock().await;

        *TEST_AGENT_RUNNER.lock() = Some(Arc::new(
            |_desc: String,
             _task: String,
             _tools: Option<Vec<String>>,
             _sys: Option<String>,
             _max: Option<u32>,
             _ctx: Arc<ToolContext>| {
                Box::pin(async move {
                    tokio::time::sleep(std::time::Duration::from_secs(
                        AGENT_TIMEOUT_SECS + 5,
                    ))
                    .await;
                    "should not reach here".to_string()
                }) as Pin<Box<dyn std::future::Future<Output = String> + Send>>
            },
        ));

        let ctx = make_ctx(0);
        let result = TeamCreateTool
            .execute(
                serde_json::json!({
                    "team_name": "test-timeout",
                    "task": "slow task",
                    "agents": [{"name": "slow-bot"}],
                    "parallel": false,
                }),
                &ctx,
            )
            .await;

        // execute itself must succeed (timeout is per-agent, not a hard error).
        assert!(
            !result.is_error,
            "execute should succeed even when an agent times out: {}",
            result.content
        );
        let val: serde_json::Value = serde_json::from_str(&result.content).unwrap();
        let output = val["results"][0]["output"].as_str().unwrap_or("");
        assert!(
            output.contains("timed out"),
            "expected timeout message in agent output, got: {}",
            output
        );

        *TEST_AGENT_RUNNER.lock() = None;
    }
}
