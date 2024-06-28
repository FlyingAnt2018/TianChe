import json
from pathlib import Path

workspace = Path.cwd()

class BaseClass:
    '''
    对文件的存在性进行检查
    '''
    def __init__(self, conf_path=""):
        # 安全检查
        assert conf_path != "", f"[Error]  conf_path is null"
        conf_path = workspace / conf_path
        assert conf_path.exists() and conf_path.is_file(), f"[Error]  File {conf_path} not found."

        # 成员获取
        self.workspace = workspace
        self.conf_path = workspace / conf_path
        print(self.conf_path)
        
        self.conf = json.loads(conf_path.read_text(encoding='utf-8'))