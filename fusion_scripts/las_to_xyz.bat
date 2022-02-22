@echo off

set fusiondir=%1
set points=%2
set output=%3

%fusiondir%LDA2ASCII %points% %output%.txt 0