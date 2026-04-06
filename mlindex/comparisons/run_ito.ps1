# Configuration
$exePath = "C:\Users\David\Documents\powder\ito13\ito13.exe"  # Replace with your .exe path
$rootDir = "C:\Users\David\Documents\powder\results\ito"    # Replace with the root directory containing subdirectories

# Get all .dat files in subdirectories
Get-ChildItem -Path $rootDir -Recurse -Filter "*.dat" | ForEach-Object {
    $datFile = $_
    $subDir = $datFile.DirectoryName

    # Change into the subdirectory so we can use just the filename
    Set-Location -Path $subDir

    Write-Host "Processing: $($datFile.BaseName)"

    # Run the command using just the filename (no path)
    $output = echo $datFile.BaseName | & $exePath

    # Save output to the subdirectory
    $outputFile = Join-Path $subDir "$($datFile.BaseName)_output.txt"
    $output | Out-File -FilePath $outputFile -Encoding utf8

    Write-Host "Output saved to: $outputFile"
}