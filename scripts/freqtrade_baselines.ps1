param(
    [string]$Config = "config/freqtrade.dryrun.example.json",
    [string]$Timerange = "20240101-20240102",
    [string]$Timeframe = "5m"
)

$ErrorActionPreference = "Stop"

$strategies = @(
    "MaCrossoverBaseline",
    "MomentumBaseline",
    "RsiMeanReversionBaseline",
    "RandomEntryBaseline",
    "BuyAndHoldBaseline"
)

foreach ($strategy in $strategies) {
    freqtrade backtesting `
        --config $Config `
        --strategy $strategy `
        --strategy-path user_data/strategies `
        --timerange $Timerange `
        --timeframe $Timeframe

    freqtrade lookahead-analysis `
        --config $Config `
        --strategy $strategy `
        --strategy-path user_data/strategies `
        --timerange $Timerange `
        --timeframe $Timeframe
}
