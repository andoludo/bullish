from huey import SqliteHuey  # type: ignore

huey = SqliteHuey(name="queue")
