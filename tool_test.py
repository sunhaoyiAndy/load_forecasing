import lwhtool

console_output = lwhtool.ConsoleOutput()


a, b = console_output.start()

help(list)
console_output.stop(a, b)

