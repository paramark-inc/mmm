import git
import logging


def git_sha() -> str:
    try:
        repo = git.Repo(search_parent_directories=True)
        return repo.head.object.hexsha
    except ValueError:
        logging.error("Couldn't get git sha, check directory ownership and safe.directory config")
