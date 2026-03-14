param(
    [Parameter(Mandatory = $true)]
    [string]$InputJson,

    [Parameter(Mandatory = $true)]
    [string]$OutputJson,

    [string]$LanguageTag = "ru"
)

$ErrorActionPreference = "Stop"

Add-Type -AssemblyName System.Runtime.WindowsRuntime
[Windows.Storage.StorageFile, Windows.Storage, ContentType = WindowsRuntime] | Out-Null
[Windows.Graphics.Imaging.BitmapDecoder, Windows.Foundation, ContentType = WindowsRuntime] | Out-Null
[Windows.Media.Ocr.OcrEngine, Windows.Foundation, ContentType = WindowsRuntime] | Out-Null
[Windows.Globalization.Language, Windows.Globalization, ContentType = WindowsRuntime] | Out-Null

function Await-WinRt {
    param(
        [Parameter(Mandatory = $true)]
        $Operation,

        [Parameter(Mandatory = $true)]
        [Type]$ResultType
    )

    $asTask = ([System.WindowsRuntimeSystemExtensions]).GetMethods() |
        Where-Object { $_.Name -eq "AsTask" -and $_.IsGenericMethod -and $_.GetParameters().Count -eq 1 } |
        Select-Object -First 1

    $task = $asTask.MakeGenericMethod($ResultType).Invoke($null, @($Operation))
    $task.Wait()
    return $task.Result
}

function Invoke-OcrText {
    param(
        [Parameter(Mandatory = $true)]
        [string]$ImagePath,

        [Parameter(Mandatory = $true)]
        [Windows.Media.Ocr.OcrEngine]$Engine
    )

    $file = Await-WinRt ([Windows.Storage.StorageFile]::GetFileFromPathAsync($ImagePath)) ([Windows.Storage.StorageFile])
    $stream = Await-WinRt ($file.OpenAsync([Windows.Storage.FileAccessMode]::Read)) ([Windows.Storage.Streams.IRandomAccessStream])
    $decoder = Await-WinRt ([Windows.Graphics.Imaging.BitmapDecoder]::CreateAsync($stream)) ([Windows.Graphics.Imaging.BitmapDecoder])
    $bitmap = Await-WinRt ($decoder.GetSoftwareBitmapAsync()) ([Windows.Graphics.Imaging.SoftwareBitmap])
    $result = Await-WinRt ($Engine.RecognizeAsync($bitmap)) ([Windows.Media.Ocr.OcrResult])
    return ($result.Text | Out-String).Trim()
}

$items = Get-Content -Path $InputJson -Raw | ConvertFrom-Json
$language = [Windows.Globalization.Language]::new($LanguageTag)
$engine = [Windows.Media.Ocr.OcrEngine]::TryCreateFromLanguage($language)

if (-not $engine) {
    throw "Failed to create OCR engine for language '$LanguageTag'."
}

$output = foreach ($item in $items) {
    $id = [string]$item.id
    $path = [string]$item.path
    try {
        [pscustomobject]@{
            id = $id
            text = (Invoke-OcrText -ImagePath $path -Engine $engine)
            error = $null
        }
    }
    catch {
        [pscustomobject]@{
            id = $id
            text = ""
            error = $_.Exception.Message
        }
    }
}

$output | ConvertTo-Json -Depth 5 | Set-Content -Path $OutputJson -Encoding utf8
