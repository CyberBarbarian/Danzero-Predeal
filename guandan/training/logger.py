import os, csv, time, json, logging
from typing import Dict, Any

class Logger:
    def __init__(self, log_dir: str, xpid: str):
        self.base = os.path.join(log_dir, xpid)
        os.makedirs(self.base, exist_ok=True)
        self.fields_path = os.path.join(self.base, 'fields.csv')
        self.logs_path = os.path.join(self.base, 'logs.csv')
        self.meta_path = os.path.join(self.base, 'meta.json')
        self.tick = 0
        self.fieldnames = ['_tick','_time']
        logging.basicConfig(level=logging.INFO)
        self._log = logging.getLogger('training.logger')

    def log(self, data: Dict[str, Any]):
        data = dict(data)
        data['_tick'] = self.tick
        data['_time'] = time.time()
        self.tick += 1
        # update fields
        new = False
        for k in data:
            if k not in self.fieldnames:
                self.fieldnames.append(k)
                new = True
        if new or self.tick == 1:
            with open(self.fields_path,'w',newline='') as f:
                csv.writer(f).writerow(self.fieldnames)
        if self.tick == 1:
            if not os.path.exists(self.logs_path):
                with open(self.logs_path,'a') as f:
                    f.write('# ' + ','.join(self.fieldnames) + '\n')
        with open(self.logs_path,'a',newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerow(data)

    def save_meta(self, meta: Dict[str, Any]):
        with open(self.meta_path,'w') as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

__all__ = ['Logger']

