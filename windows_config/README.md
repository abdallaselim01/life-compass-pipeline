# windows_config

Everything that belongs to the system but has to live inside Windows itself (Task Scheduler and Startup). The backup copies are kept here so the whole project stays in this folder.

- `scheduled_tasks\` - an export of the 4 scheduled tasks (Retry Pending, Random Resurface, Listener Watchdog, Health Check). The easiest way to restore them is `python scripts\install_tasks.py` (it registers the missing ones and enables the disabled ones; the Health Check does the same by itself every hour). The saved XML carries placeholder paths (`C:\LifeCompass`, `C:\Python\pythonw.exe`) and no user id: when a task is registered, the working directory, the script path and the Python are replaced with the current ones, and the task runs as the current user. The files are UTF-8 although their first line says UTF-16.
  Manual restore from PowerShell (the paths are NOT localized this way, so prefer the script above):
  `Register-ScheduledTask -TaskName "LifeCompass Retry Pending" -Xml (Get-Content -Raw scheduled_tasks\task_LifeCompass_Retry_Pending.xml)`
  (the same command for the other tasks; if the task already exists, remove it first with `Unregister-ScheduledTask`).
- **Startup**: a `LifeCompass listener.lnk` shortcut in the Windows Startup folder runs `start_listener_silent.vbs` from the project folder (there is no copy of the launcher outside the project folder).
