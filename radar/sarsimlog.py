# -*- coding: utf-8 -*-
"""
Created on Wed Apr 14 15:50:46 2021

@author: SIKANETAI
"""

#%%
import logging
import logging.config
import logging.handlers
import os
import json
from datetime import datetime
filepath = r"C:\Users\ishuwa.sikaneta\OneDrive - ESA\Documents\ESTEC\Envision\ROIs\incidence"
logfile = os.path.join(filepath, "processing.log")

#%% To lowercase
logging.addLevelName(logging.DEBUG, 'debug')
logging.addLevelName(logging.INFO, 'info')
logging.addLevelName(logging.WARNING, 'warning')
logging.addLevelName(logging.ERROR, 'error')
    
#%%    
class ecsMsg:
    def __init__(self, message, level, args, kwargs):
        self.message = message
        self.kwargs = kwargs
        self.level = level

    def __str__(self):
        mesgDict = {"@timestamp": datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z',
                    "log.level": logging._levelToName[self.level],
                    "message": self.message,
                    "ecs":{
                        "version": "1.6.0"
                        },
                    "user": {"name": os.getlogin()}
                    }
            
        mesgDict.update(**self.kwargs)
        
        return json.dumps(mesgDict, separators=(',',':'))

#%%
class CustomAdapter(logging.LoggerAdapter):
    def __init__(self, logger):
        # super(LoggerAdapter, self).__init__(logger, {})
        self.logger = logger
        
    def log(self, level, msg, *args, **kwargs):
        if self.isEnabledFor(level):
            self.logger._log(level, 
                             ecsMsg(msg, 
                                    level,
                                    args, 
                                    kwargs), 
                             ())


            
#%% Logging dictionary
logconfig = {
    'version': 1,
    'disable_existing_loggers': True,
    'formatters': {
        'ecs': {
            'class': 'logging.Formatter',
            'format': '%(message)s'
        }
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'formatter': 'ecs',
            'level': 'INFO'
        },
        'rotatingFileHandler': {
            'class': 'logging.handlers.RotatingFileHandler',
            'formatter': 'ecs',
            'level': 'INFO',
            'filename': logfile,
            'mode': 'w',
            'backupCount': 10,
            'maxBytes': 50000
        },
        'staticFileHandler': {
            'class': 'logging.FileHandler',
            'formatter': 'ecs',
            'level': 'INFO',
            'filename': logfile,
            'mode': 'w'
        }
    },
    'loggers': {
        'foo': {
            # 'handlers': ['console', 'fileHandler']
            'handlers': ['console', 'staticFileHandler']
        }
    },
    'root': {
            'handlers': [],
            'level': 'DEBUG'
    }
}    

#%%
logging.config.dictConfig(logconfig)
logger = CustomAdapter(logging.getLogger('foo'))
