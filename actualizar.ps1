# Copia el codigo nuevo a C:\Sports-Prediction-Center, entrena el modelo de
# Over/Under, genera las predicciones de hoy y manana y reinicia el servidor.
# No toca .env, la base de datos ni los modelos ya entrenados.
$src = $PSScriptRoot
$dst = "C:\Sports-Prediction-Center"
# SOCCER y CHAT faltaban en esta lista: el codigo nuevo se quedaba sin copiar y
# habia que moverlo a mano. SOCCER se excluye por carpetas de datos (igual que
# NBA y TENIS) para no pisar su base ni sus modelos entrenados.
foreach ($d in "MLB\engine", "MLB\tests", "MLB\features", "MLB\ingest", "MLB\database",
              "MLB\backtests", "NFL", "NBA", "TENIS", "SOCCER", "PARLAY", "CHAT",
              "dashboard", "shared", "tests", "audit") {
    if (Test-Path "$src\$d") {
        # NUNCA se copian bases de datos ni modelos sobre los del usuario (salvo NBA la primera vez)
        $conDatos = @("NBA", "TENIS", "SOCCER") -contains $d
        $primeraVez = ($d -eq "NBA" -and -not (Test-Path "$dst\NBA\database\nba_markets.sqlite3")) -or
                      ($d -eq "TENIS" -and -not (Test-Path "$dst\TENIS\database\tenis_markets.sqlite3")) -or
                      ($d -eq "SOCCER" -and -not (Test-Path "$dst\SOCCER\database\soccer_markets.sqlite3"))
        if ($conDatos -and -not $primeraVez) {
            # Se excluye por PATRON DE ARCHIVO, no por carpeta. `models/` y
            # `features/` guardan artefactos .joblib/.parquet pero TAMBIEN son
            # paquetes de Python: excluir la carpeta entera dejaba fuera
            # cards.py y corners.py y el modulo no importaba.
            robocopy "$src\$d" "$dst\$d" /E /NFL /NDL /NJH /NJS /XD node_modules __pycache__ out `
                /XF *.sqlite3 *.sqlite3-shm *.sqlite3-wal *.joblib *.parquet | Out-Null
        } elseif ($conDatos) {
            robocopy "$src\$d" "$dst\$d" /E /NFL /NDL /NJH /NJS /XD node_modules __pycache__ | Out-Null
        } else {
            robocopy "$src\$d" "$dst\$d" /E /NFL /NDL /NJH /NJS /XD node_modules __pycache__ /XF *.sqlite3 *.sqlite3-shm *.sqlite3-wal *.joblib | Out-Null
        }
    }
}
foreach ($f in "spc.py", "README.md", "FINAL_AUDIT_REPORT.md", "NFL_MARKETS_REPORT.md",
                "NBA_MARKETS_REPORT.md", "TENIS_MARKETS_REPORT.md", "SOCCER_MARKETS_REPORT.md",
                "VALUE_AND_CALIBRATION_REPORT.md", "pytest.ini") {
    if (Test-Path "$src\$f") { Copy-Item "$src\$f" "$dst\$f" -Force }
}
# El frontend hay que recompilarlo: el navegador sirve dist/, no src/.
if (Test-Path "$src\dashboard\frontend\package.json") {
    Write-Host "== compilando el dashboard =="
    Push-Location "$dst\dashboard\frontend"
    cmd /c "npm run build" 2>&1 | Select-Object -Last 3
    Pop-Location
}
Set-Location $dst
$env:PYTHONIOENCODING = "utf-8"
$py = "$dst\.venv\Scripts\python.exe"
if (-not (Test-Path "$dst\MLB\models\total_v2.joblib")) {
    Write-Host "== entrenando modelo Over/Under MLB (1-3 min) =="
    & $py MLB\engine\train_production.py --markets total 2>&1 | Select-String -NotMatch "Warning" | Select-Object -Last 4
    & $py spc.py mlb-predict 2>&1 | Select-String -NotMatch "Warning" | Select-Object -Last 3
}
if (-not (Test-Path "$dst\NFL\models\total_v1.joblib")) {
    Write-Host "== NFL total/spread: investigacion walk-forward + entrenamiento (3-6 min) =="
    & $py spc.py nfl-train 2>&1 | Select-String -NotMatch "Warning" | Select-Object -Last 12
}
Write-Host "== NFL: cuotas + predicciones =="
& $py spc.py nfl-cycle 2>&1 | Select-String -NotMatch "Warning" | Select-Object -Last 6
Write-Host "== NBA: ciclo (base, modelos e investigacion vienen ya en el paquete) =="
& $py spc.py nba-cycle 2>&1 | Select-String -NotMatch "Warning" | Select-Object -Last 8
if (-not (Test-Path "$dst\TENIS\markets\out\features.parquet")) {
    Write-Host "== TENIS: construyendo features desde la base (2-4 min, solo la primera vez) =="
    & $py -c "from TENIS.markets.features import build; build(progress=print)" 2>&1 | Select-String -NotMatch "Warning" | Select-Object -Last 2
}
Write-Host "== TENIS: calendario, cuotas y predicciones =="
& $py spc.py tenis-cycle 2>&1 | Select-String -NotMatch "Warning" | Select-Object -Last 8
Write-Host "== combinadas =="
& $py spc.py parlays 2>&1 | Select-String -NotMatch "Warning" | Select-Object -Last 3
Write-Host "== tests =="
& $py -m pytest -q 2>&1 | Select-Object -Last 1
Write-Host "== reiniciando servidor =="
Get-NetTCPConnection -LocalPort 8100 -State Listen -ErrorAction SilentlyContinue |
    ForEach-Object { Stop-Process -Id $_.OwningProcess -Force }
Start-Sleep 2
Start-Process -FilePath $py -ArgumentList "spc.py", "serve" -WorkingDirectory $dst -WindowStyle Minimized
Start-Sleep 8
try { (Invoke-RestMethod http://127.0.0.1:8100/api/status).auto_score | ConvertTo-Json -Compress }
catch { Write-Host "el servidor aun esta arrancando; abre http://127.0.0.1:8100 en unos segundos" }
Write-Host "Listo. Recarga http://127.0.0.1:8100"
