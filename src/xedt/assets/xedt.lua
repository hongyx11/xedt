-- Neovim 0.11+. Call setup() after your other clangd configuration.
local M = {}

function M.command(dispatchers, config)
  local root = config.root_dir
  local path = root and (root .. "/.xedt/state.json")
  local state
  if path and vim.fn.filereadable(path) == 1 then
    local ok, decoded = pcall(vim.json.decode, table.concat(vim.fn.readfile(path), "\n"))
    if ok then state = decoded end
  end
  local cmd
  if state and state.schema == 1 then
    cmd = { state.clangd, "--background-index", "--enable-config=false",
      "--compile-commands-dir=" .. vim.fn.fnamemodify(state.database, ":h") }
  else
    cmd = { vim.env.VIM_CLANGD or "clangd", "--background-index" }
  end
  return vim.lsp.rpc.start(cmd, dispatchers, { cwd = root })
end

function M.setup()
  if not vim.lsp.config or not vim.lsp.enable then
    error("xedt needs Neovim 0.11+ (vim.lsp.config/enable)")
  end
  vim.filetype.add({ extension = { cu = "cuda", cuh = "cuda" } })
  vim.lsp.config("clangd", {
    cmd = M.command,
    filetypes = { "c", "cpp", "objc", "objcpp", "cuda" },
    root_markers = { ".xedt.json", "compile_commands.json", ".clangd", ".git" },
  })
  vim.lsp.enable("clangd")
end

return M
