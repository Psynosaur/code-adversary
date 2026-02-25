import type { Plugin } from "@opencode-ai/plugin";

/**
 * Code Reviewer Enforcement Plugin
 *
 * Hooks into OpenCode's lifecycle to enforce adversarial code review.
 *
 * Trigger: when long-term-memory_remember is called AFTER file edits
 * have been made in the session. This means the agent considers the
 * work done and is storing findings -- time to review.
 *
 * Does NOT trigger on every file edit. The memory store is the signal.
 */
export const CodeReviewerPlugin: Plugin = async ({
  client,
  directory,
}) => {
  const projectName = directory.split("/").pop() || "unknown";

  // Per-session state
  const sessionState = new Map<
    string,
    {
      filesEdited: Set<string>;
      reviewCalled: boolean;
      memoryStoreCount: number;
      reviewReminded: boolean;
    }
  >();

  const getState = (sessionID: string) => {
    if (!sessionState.has(sessionID)) {
      sessionState.set(sessionID, {
        filesEdited: new Set(),
        reviewCalled: false,
        memoryStoreCount: 0,
        reviewReminded: false,
      });
    }
    return sessionState.get(sessionID)!;
  };

  const log = (message: string, extra: Record<string, any> = {}) => {
    client.app
      .log({
        body: {
          service: "code-reviewer",
          level: "info",
          message,
          extra: { projectName, ...extra },
        },
      })
      .catch(() => {});
  };

  log("Plugin initialized", { directory });

  return {
    /**
     * HOOK 1: System Prompt — inject review instructions
     *
     * Tells the agent about the review workflow so it knows to call
     * review_diff after completing work.
     */
    "experimental.chat.system.transform": async (_input, output) => {
      output.system.push(`## Adversarial Code Review

You have access to an adversarial code reviewer via the \`code-reviewer\` MCP server.

### Review Workflow
After you finish making changes and store your findings in long-term-memory:
1. Run \`git diff\` to capture all changes made in this session
2. Call \`code-reviewer_review_diff\` with:
   - \`diff\`: the git diff output
   - \`memories\`: JSON string of the memories you just stored (for context)
3. Address any **critical** findings before considering the task complete
4. Store the review findings in long-term-memory with tag "review,${projectName}"

### When to Review
- After completing a feature or bug fix (when you store the work summary in memory)
- After refactoring (when you store the decision/outcome in memory)
- Do NOT review trivial changes (typo fixes, comment-only edits)

### Available Review Tools
- \`code-reviewer_review_diff\` — adversarial review of git diff (primary tool)
- \`code-reviewer_review_pattern\` — scrutinize a specific code pattern
- \`code-reviewer_challenge_decision\` — stress-test an architectural decision
- \`code-reviewer_get_persona\` — see the reviewer's adversarial traits`);
    },

    /**
     * HOOK 2: Track file edits and review tool calls
     */
    "tool.execute.after": async (input, _output) => {
      if (!input.sessionID || !input.tool) return;
      const state = getState(input.sessionID);

      // Track file edits
      if (input.tool === "edit" || input.tool === "write") {
        const filePath = input.args?.filePath;
        if (filePath && typeof filePath === "string") {
          state.filesEdited.add(filePath);
          log("File edited", {
            sessionID: input.sessionID,
            file: filePath,
            totalEdited: state.filesEdited.size,
          });
        }
      }

      // Track review calls
      if (
        input.tool === "code-reviewer_review_diff" ||
        input.tool === "code-reviewer_review_pattern"
      ) {
        state.reviewCalled = true;
        log("Review completed", {
          sessionID: input.sessionID,
          tool: input.tool,
          filesEdited: state.filesEdited.size,
        });
      }
    },

    /**
     * HOOK 3: Intercept memory stores — trigger review reminder
     *
     * When long-term-memory_remember is called and files have been
     * edited but no review has been done yet, inject a reminder.
     */
    "tool.execute.before": async (input, _output) => {
      if (!input.sessionID || !input.tool) return;
      if (input.tool !== "long-term-memory_remember") return;

      const state = getState(input.sessionID);
      state.memoryStoreCount++;

      // Only remind once per session, and only if files were actually edited
      if (
        state.filesEdited.size > 0 &&
        !state.reviewCalled &&
        !state.reviewReminded
      ) {
        state.reviewReminded = true;
        log("Review reminder triggered", {
          sessionID: input.sessionID,
          filesEdited: state.filesEdited.size,
          memoryStores: state.memoryStoreCount,
        });
      }
    },

    /**
     * HOOK 4: Compaction — preserve review state
     */
    "experimental.session.compacting": async (input, output) => {
      if (!input.sessionID) return;
      const state = getState(input.sessionID);

      if (state.filesEdited.size === 0) return;

      const filesList = Array.from(state.filesEdited).slice(0, 20).join(", ");

      output.context.push(`## Code Review Status

Files edited this session (${state.filesEdited.size}): ${filesList}
Review completed: ${state.reviewCalled ? "Yes" : "No"}
Memory stores: ${state.memoryStoreCount}

${
  !state.reviewCalled
    ? `IMPORTANT: Files were edited but no adversarial review was performed.
After compaction, run \`git diff\` and call \`code-reviewer_review_diff\` to review changes.
Store review findings in long-term-memory with tag "review,${projectName}".`
    : "Review was completed this session. No further review action needed unless new changes are made."
}`);
    },

    /**
     * HOOK 5: Session cleanup
     */
    event: async ({ event }) => {
      if (event.type === "session.deleted") {
        const props = (event as any).properties || {};
        if (props.id) {
          sessionState.delete(props.id);
        }
      }
    },
  };
};
