# -*- coding:utf-8 -*-
# Author：Leslie Dang
# Initial Data : 2019/11/11 14:13

#!/usr/local/bin/python
# -*- coding:utf-8 -*-

import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

logger.info("Start print log")
logger.debug("Do something")
logger.warning("Something maybe fail.")
logger.info("Finish")