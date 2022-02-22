@echo off

set fusiondir=%1
set points=%2
set output=%3
set xmin=%4
set xmax=%5
set ymin=%6
set ymax=%7
set classes=%8

%fusiondir%clipdata /class:%classes% %points% %output%.las %xmin% %xmax% %ymin% %ymax%
%fusiondir%LDA2ASCII %output%.las %output%.txt 0