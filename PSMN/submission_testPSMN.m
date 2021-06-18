function submission_testPSMN(plotornot)
%----------------------------------------------------------------------------
%How to use:
%Enter your path in the session path variable (2 fields: session.input_path
% and session.output_path),
%Enter your manip name in the ManipName variable
%Compile the file with the following command: mcc -m submission_matlab.m
%You will obtain several file, 2 are usefull run_submission_matlab.sh and
%submision_matlab.exe
%IMPORTANT: in run_submission_matlab.sh, replace the line 30 by
%  eval "/Yourpath/submission_matlab" $args
%Once you did this you can modified and call the file
%submision_%tracking.sh in a terminal to launch the parallelisation
plotornot=PlotOrNot;
varOut=testPSMN(plotornot)
end
