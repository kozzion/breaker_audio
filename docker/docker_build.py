
command =  'docker build'
command += ' -t image_service_synthesize'
command += ' -f dockerfile_service_synthesize'
# command += '--build-arg SSH_PRIVATE_KEY="' + string_git_key + ''
command += ' build'


import os
print(command)
os.system(command)