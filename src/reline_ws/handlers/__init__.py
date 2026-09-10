"""Protocol methods, one module per area: `run` (start/stop), `fs` (ls),
`configs` (preset files). Each handler answers the request it was given and
never touches the socket directly — `Connection` owns transport and state."""
