param(
    [string]$Workspace = "D:\trabajo_PLN\Trabajo_PLN-MIA-Grupo4",
    [string]$DriveBundle = "G:\My Drive\PLN_colab_04_artifacts"
)

$script = Join-Path (Split-Path -Parent $MyInvocation.MyCommand.Path) "sincronizar_04_20x_google_drive.ps1"
& $script -Workspace $Workspace -DriveBundle $DriveBundle
exit $LASTEXITCODE
