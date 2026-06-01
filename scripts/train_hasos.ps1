param(
    [int]$Gpu = -1,
    [int]$Epochs = 10,
    [string]$RunId = "hasos_run1"
)

$ErrorActionPreference = "Stop"
Push-Location $PSScriptRoot
try {
    python .\prepare_hasos.py
    python ..\src\utils\train-spm.py ..\data\hasos\hasos_summ.json ..\data\sentencepiece\hasos_unigram_32k
    python ..\src\train.py `
        --data ..\data\hasos\hasos_summ.json `
        --sentencepiece ..\data\sentencepiece\hasos_unigram_32k.model `
        --run_id $RunId `
        --gpu $Gpu `
        --epochs $Epochs
}
finally {
    Pop-Location
}
