print("Loading .nvim.lua started")
local dap = require("dap")
dap.adapters.python = {
	type = "executable",
	command = "python",
	args = { "-m", "debugpy.adapter" },
}

dap.configurations.python = {
	{
		type = "python",
		request = "launch",
		name = "Launch Trans",
		program = "${file}",
		args = { "--model_name", "trans" },
	},
	{
		type = "python",
		request = "launch",
		name = "Launch MLP",
		program = "${file}",
	},
}
print("Loading .nvim.lua completed")
