@echo off
cd /d "%~dp0"
Rscript -e "shiny::runApp('app_shiny.R', launch.browser = TRUE)"
