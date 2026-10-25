"""Harbor import target: `guardfork.fork.fork_agent:ForkReplayAppWorldAgent`.

Importable only inside the Harness-Zero environment (PYTHONPATH with harness-zero root/src).
"""

from guardfork.fork.harbor_replay import make_fork_agent_class

ForkReplayAppWorldAgent = make_fork_agent_class()
