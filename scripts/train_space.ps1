param(
    [int]$Gpu = 0,
    [int]$Epochs = 20,
    [string]$RunId = "space_run1"
)

$ErrorActionPreference = "Stop"
Push-Location $PSScriptRoot
try {
    python ..\src\train.py `
        --data ..\data\space\json\space_train.json `
        --sentencepiece ..\data\sentencepiece\space_unigram_32k.model `
        --run_id $RunId `
        --gpu $Gpu `
        --epochs $Epochs
}
finally {
    Pop-Location
}
