Set WshShell = CreateObject("WScript.Shell")
Set fso = CreateObject("Scripting.FileSystemObject")
' The project folder is resolved from this file's own location, so the
' project can be moved without editing this launcher (a hardcoded
' path would silently break the watchdog's relaunch after a move).
' PYTHONPATH points at src\ so the package is importable straight from the
' project folder -- no pip "editable" install (which would write files into
' site-packages, outside the project).
root = fso.GetParentFolderName(WScript.ScriptFullName)
' listener.log over 5 MB is MOVED (never deleted) to logs\archive before the new listener starts: the old listener is gone by now,
' so the file is free. If anything holds it, the move is simply skipped (On Error Resume Next) and the next launch tries again.
On Error Resume Next
logPath = root & "\listener.log"
If fso.FileExists(logPath) Then
    If fso.GetFile(logPath).Size > 5242880 Then
        If Not fso.FolderExists(root & "\logs\archive") Then fso.CreateFolder root & "\logs\archive"
        n = Now
        stamp = Year(n) & Right("0" & Month(n), 2) & Right("0" & Day(n), 2) & "_" & Right("0" & Hour(n), 2) & Right("0" & Minute(n), 2) & Right("0" & Second(n), 2)
        fso.MoveFile logPath, root & "\logs\archive\listener_" & stamp & ".log"
    End If
End If
On Error GoTo 0
WshShell.Run "cmd /c cd /d """ & root & """ && set ""PYTHONPATH=" & root & "\src"" && python -B -u -m lifecompass.listener >> listener.log 2>&1", 0, False
