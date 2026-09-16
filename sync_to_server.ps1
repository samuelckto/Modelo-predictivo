# ==============================================================================
# Sports Prediction Center - Migracion de Datos y Modelos a DietPi
# ==============================================================================
param (
    [Parameter(Mandatory=$false)]
    [string]$ServerIp = "100.77.2.80",
    
    [Parameter(Mandatory=$false)]
    [string]$User = "root",

    [Parameter(Mandatory=$false)]
    [string]$RemoteDir = "/root/Sports-Prediction-Center"
)

$ErrorActionPreference = "Stop"

Write-Host "=================================================================" -ForegroundColor Cyan
Write-Host " Sincronizando bases de datos e historico hacia el servidor DietPi" -ForegroundColor Cyan
Write-Host " Destino: ${User}@${ServerIp}:${RemoteDir}" -ForegroundColor Cyan
Write-Host "=================================================================" -ForegroundColor Cyan

# 1. Asegurar directorios remotos
Write-Host "[1/4] Creando estructura de directorios en el servidor..." -ForegroundColor Yellow
ssh "${User}@${ServerIp}" "mkdir -p ${RemoteDir}/MLB/database ${RemoteDir}/NBA/database ${RemoteDir}/TENIS/database ${RemoteDir}/TENIS/markets/out ${RemoteDir}/SOCCER/database ${RemoteDir}/NFL/database ${RemoteDir}/PARLAY/database"

# 2. Copiar archivo .env (con credenciales privadas que no van a Git)
if (Test-Path ".env") {
    Write-Host "[2/4] Copiando .env con credenciales..." -ForegroundColor Yellow
    scp ".env" "${User}@${ServerIp}:${RemoteDir}/.env"
}

# 3. Copiar archivo de features de tenis (>100MB)
$tenisFeatures = "TENIS\markets\out\features.parquet"
if (Test-Path $tenisFeatures) {
    Write-Host "[3/4] Transfiriendo $tenisFeatures hacia DietPi (146 MB)..." -ForegroundColor Yellow
    scp $tenisFeatures "${User}@${ServerIp}:${RemoteDir}/TENIS/markets/out/features.parquet"
}

# 4. Copiar bases de datos SQLite locales existentes
Write-Host "[4/4] Transfiriendo bases de datos SQLite existentes..." -ForegroundColor Yellow
$dbs = @(
    "MLB\database\mlb.sqlite3",
    "NBA\database\nba_markets.sqlite3",
    "TENIS\database\tenis_markets.sqlite3",
    "SOCCER\database\soccer_markets.sqlite3",
    "NFL\database\nfl_markets.sqlite3",
    "PARLAY\database\parlays.sqlite3"
)

foreach ($db in $dbs) {
    if (Test-Path $db) {
        $remotePath = "${RemoteDir}/" + ($db -replace '\\', '/')
        Write-Host "  -> Copiando $db..." -ForegroundColor Gray
        scp $db "${User}@${ServerIp}:${remotePath}"
    }
}

Write-Host ""
Write-Host "=================================================================" -ForegroundColor Green
Write-Host " ¡Sincronizacion completada con exito!" -ForegroundColor Green
Write-Host " Todos los datos historicos y credenciales estan en tu DietPi." -ForegroundColor Green
Write-Host " Reinicia el servicio para cargar los datos: ssh ${User}@${ServerIp} 'systemctl restart spc.service'" -ForegroundColor Green
Write-Host "=================================================================" -ForegroundColor Green
