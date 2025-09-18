import os
from datetime import datetime
try:
    from git import Repo
    repo = Repo(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    branch = repo.active_branch.name
    last_commit = repo.head.commit
    __version__ = branch + ':' + str(last_commit)
except:
    now = datetime.now()
    formatted = now.strftime("%Y-%m-%d %H:%M:%S")
    __version__ = formatted