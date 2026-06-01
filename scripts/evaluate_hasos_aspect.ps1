param(
    [Parameter(Mandatory = $true)]
    [string]$Model,
    [int]$Gpu = -1,
    [string]$RunId = "hasos_aspects_run1",
    [int]$MaxTokens = 40
)

$ErrorActionPreference = "Stop"
Push-Location $PSScriptRoot
try {
    python .\prepare_hasos.py
    $aspects = (Get-Content ..\data\hasos\aspect_taxonomy.tsv | Select-Object -Skip 1 | ForEach-Object { ($_ -split "`t")[1] }) -join ","
    python ..\src\aspect_inference.py `
        --summary_data ..\data\hasos\hasos_summ.json `
        --sentencepiece ..\data\sentencepiece\hasos_unigram_32k.model `
        --seedsdir ..\data\seeds_hasos `
        --gold_aspects $aspects `
        --model $Model `
        --run_id $RunId `
        --gpu $Gpu `
        --max_tokens $MaxTokens `
        --no_eval
}
finally {
    Pop-Location
}
