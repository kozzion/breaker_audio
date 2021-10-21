import pickle
import json
from pathlib import Path
from breaker_core.datasource.bytessource_file import BytessourceFile
path = Path('C:\\project\\data\\data_breaker\\breaker_discord\\bot_dev\\response\\voice_authenticator\\b6de8579837e88034e788d1ae8e831147b5b32414cc1d5877c50cab39490ae4d')
with path.open('rb') as file:
    report = json.load(file)
print(report)

import matplotlib.pyplot as plt 
plt.figure()
plt.hist(report['authentication_report']['list_val_a_b'], bins=20)
plt.hist(report['authentication_report']['list_val_b_b'], bins=20)
plt.title('similarity')
plt.ylabel('count')
plt.xlabel('similarity')
plt.legend(['attempt', 'reference'])
plt.show()