@echo off

set fusiondir=%1
set filelist=%2
set output=%3

%fusiondir%MergeData %filelist% %output%