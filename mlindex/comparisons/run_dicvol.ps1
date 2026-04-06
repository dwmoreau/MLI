# Configuration
$exePath = "C:\Users\David\Documents\powder\dicvol91\DICVOL91.exe"  # Replace with your .exe path
$rootDir = "C:\Users\David\Documents\powder\results\dicvol"    # Replace with the root directory containing subdirectories
$timeoutSeconds = 120                       # Replace with your desired timeout in seconds

# Get all .dat files in subdirectories
Get-ChildItem -Path $rootDir -Recurse -Filter "*.dat" | ForEach-Object {
    $datFile = $_
    $subDir = $datFile.DirectoryName

    # Check if a corresponding .imp file already exists
    $impFile = Join-Path -Path $subDir -ChildPath "$($datFile.BaseName).imp"
    if (Test-Path $impFile) {
        Write-Host "Skipping: $($datFile.BaseName) (.imp file already exists)"
        return  # Skip to next file in ForEach-Object
    }

    Set-Location -Path $subDir

    Write-Host "Processing: $($datFile.BaseName)"

    # Use cmd /c to run the echo pipe, which better emulates interactive input
    $process = Start-Process -FilePath "cmd.exe" `
        -ArgumentList "/c echo $($datFile.BaseName) | `"$exePath`"" `
        -RedirectStandardOutput "$($datFile.BaseName)_output.txt" `
        -NoNewWindow -PassThru

    $finished = $process.WaitForExit($timeoutSeconds * 1000)

    if (-not $finished) {
        Write-Host "Timeout reached for $($datFile.BaseName), killing process..."
        $process.Kill()
    } else {
        Write-Host "Output saved to: $($datFile.BaseName)_output.txt"
    }
}