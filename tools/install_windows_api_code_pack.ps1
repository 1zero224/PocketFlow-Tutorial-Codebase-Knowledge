param(
    [string]$Version = "1.1.0",
    [string]$Destination = (Join-Path $PSScriptRoot "..\\webapp\\bin")
)

$ErrorActionPreference = "Stop"

Add-Type -AssemblyName System.IO.Compression.FileSystem

$packages = @(
    @{
        Url = "https://api.nuget.org/v3-flatcontainer/microsoft.windowsapicodepack.core/$Version/microsoft.windowsapicodepack.core.$Version.nupkg"
        Entry = "lib/Microsoft.WindowsAPICodePack.dll"
        Output = "Microsoft.WindowsAPICodePack.dll"
    },
    @{
        Url = "https://api.nuget.org/v3-flatcontainer/microsoft.windowsapicodepack.shell/$Version/microsoft.windowsapicodepack.shell.$Version.nupkg"
        Entry = "lib/Microsoft.WindowsAPICodePack.Shell.dll"
        Output = "Microsoft.WindowsAPICodePack.Shell.dll"
    }
)

function Copy-ZipEntryToFile {
    param(
        [Parameter(Mandatory = $true)]
        [string]$ZipPath,
        [Parameter(Mandatory = $true)]
        [string]$EntryPath,
        [Parameter(Mandatory = $true)]
        [string]$OutputPath
    )

    $archive = [System.IO.Compression.ZipFile]::OpenRead($ZipPath)
    try {
        $entry = $archive.GetEntry($EntryPath)
        if ($null -eq $entry) {
            throw "压缩包中未找到条目: $EntryPath"
        }

        $entryStream = $entry.Open()
        try {
            $outputStream = [System.IO.File]::Create($OutputPath)
            try {
                $entryStream.CopyTo($outputStream)
            }
            finally {
                $outputStream.Dispose()
            }
        }
        finally {
            $entryStream.Dispose()
        }
    }
    finally {
        $archive.Dispose()
    }
}

$resolvedDestination = [System.IO.Path]::GetFullPath($Destination)
New-Item -ItemType Directory -Force -Path $resolvedDestination | Out-Null

$tempRoot = Join-Path ([System.IO.Path]::GetTempPath()) ("windowsapicodepack-" + [System.Guid]::NewGuid())
New-Item -ItemType Directory -Force -Path $tempRoot | Out-Null

try {
    foreach ($package in $packages) {
        $packageFile = Join-Path $tempRoot ([System.IO.Path]::GetFileName($package.Url))
        Invoke-WebRequest -Uri $package.Url -OutFile $packageFile
        $outputPath = Join-Path $resolvedDestination $package.Output
        Copy-ZipEntryToFile -ZipPath $packageFile -EntryPath $package.Entry -OutputPath $outputPath
        Write-Output "Installed $($package.Output) -> $outputPath"
    }
}
finally {
    Remove-Item -LiteralPath $tempRoot -Recurse -Force -ErrorAction SilentlyContinue
}
