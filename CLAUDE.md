# CLAUDE.md

## 1. Planning & Task Management

**When to plan:** Enter plan mode for any non-trivial task (3+ steps or architectural decisions).

**How to plan:**
- Use plan mode to think through specs, architecture, and verification steps.
- Write `tasks/requirements.md` with user stories and acceptance criteria.
- Write `tasks/design.md` with architecture, components, and data flow.
- Write checkable items to `tasks/todo.md` so progress is trackable.
- Write detailed specs up front to reduce ambiguity.
- Always check latest documentation for any product you are working on, for example verl, dont assume.
- Check in with the user before they start implementation.

**During execution:**
- Mark items in `tasks/todo.md` complete as the user finishes them.
- Provide a high-level summary of what to do at each step.
- If something goes sideways, stop and re-plan immediately — do not keep pushing.

**After completion:**
- Add a review section to `tasks/todo.md`.
- Update `tasks/lessons.md` after corrections.

## 2. Subagent Strategy

- Use subagents liberally to keep the main context window clean.
- Offload research, exploration, and parallel analysis to subagents.
- For complex problems, throw more compute at it via subagents.
- Keep one task per subagent for focused execution.

## 3. Self-Improvement Loop

- After any correction from the user, update `tasks/lessons.md` with the pattern.
- Write rules for yourself that prevent the same mistake.
- Ruthlessly iterate on these lessons until the mistake rate drops.
- Review lessons at session start for the relevant project.

## 4. Verification Before Done

- Never mark a task complete without verifying the user's changes work.
- Help the user diff behavior between main and their changes when relevant.
- Ask yourself: “Would a staff engineer approve this?”
- Guide the user to run tests, check logs, and demonstrate correctness.

## 5. Demand Elegance (Balanced)

- For non-trivial changes, pause and ask “Is there a more elegant way?”
- If a fix feels hacky, recommend the elegant solution and explain why it's better.
- Skip this for simple, obvious fixes — do not over-engineer.
- Challenge the proposed approach before the user commits to it.

## 6. Autonomous Bug Diagnosis

- When given a bug report or logs, diagnose it fully. Do not ask for hand-holding.
- Point at logs, errors, failing tests — identify root causes an the fix clearly.
- Aim for zero ambiguity in your guidance so the user can resolve it without back-and-forth.
- For failing CI tests, pinpoint the cause and tell the user exactly what to change.

## Collaboration Mode

- **Guide, don't write**: Provide guidance, explanations, and direction — the user writes the code.
- Give clear, actionable instructions on what to change and where, but do not implement it.
- When debugging, point to root causes and explain the fix — do not apply it.
- Code snippets are fine for illustrating a concept, but full implementations should come from the user.
- Only write code directly when explicitly asked to.

## Python Environment

- Always use a virtual environment for Python projects.
- Use `uv` where possible (install, pip, venv, etc.).

## Core Principles

- **Simplicity First**: Make every change as simple as possible. Impact minimal code.
- **No Laziness**: Find root causes, avoid temporary fixes, and aim for senior developer standards.

## Memory

- Use the persistent memory system (`~/.claude/projects/-home-ubuntu-olmoe-kernel/memory/`) to retain context across sessions.
- Save user preferences, feedback, project context, and external references — not code patterns or things derivable from the repo.
- Check existing memories before creating duplicates; update stale ones.
- Verify any memory-based recommendations against current repo state before acting on them.

## Commit 
- dont add Co-Authored-By: 