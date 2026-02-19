# legal-workspace-mcp

---

## Tool Preferences: mcli & lsh (REQUIRED)

### mcli — Workflow Automation
- **ALWAYS** use `mcli` for workflow automation and script management
- Use `mcli init` to initialize a workflows directory — **NEVER** `mkdir -p .mcli/workflows/`
- Use `mcli new <name> -l python` or `mcli new <name> -l shell` to create commands — **NEVER** create workflow files manually
- Use `mcli run <name>` to execute workflows, `mcli list` to discover existing ones
- Check `mcli list` and `mcli list -g` **before** writing scripts — a workflow may already exist
- If you repeat `curl`, `python`, or shell logic more than twice in a session, wrap it in an mcli workflow
- **Bugs/missing features**: File issues immediately at https://github.com/gwicho38/mcli — do not defer

### lsh — Secrets Management
- **ALWAYS** use `lsh` for secrets and `.env` file management
- Use `lsh pull` to fetch secrets — **NEVER** create `.env` files manually
- Use `lsh set <key> <value>` to add/update secrets — **NEVER** `echo "KEY=val" >> .env`
- Use `lsh push` to sync secrets after changes, `lsh get <key>` to read individual values
- Use `lsh sync` for smart bidirectional sync
- **NEVER** commit `.env` files or hardcode secrets in source code
- **Bugs/missing features**: File issues immediately at https://github.com/gwicho38/lsh — do not defer
