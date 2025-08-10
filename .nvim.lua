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
		name = "Launch",
		program = "${file}",
	},

	{
		type = "python",
		request = "launch",
		name = "Launch with Args",
		program = "${file}",
		args = { "--model_name", "trans" },
	},
}
print("Loading .nvim.lua completed")
