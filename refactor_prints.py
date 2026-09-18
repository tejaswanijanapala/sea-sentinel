import os
import glob
import re

def insert_imports(content):
    if 'get_logger' in content:
        return content
        
    import_statement = 'from shared.utils.logger import get_logger\nlogger = get_logger(__name__)\n\n'
    
    lines = content.split('\n')
    insert_idx = 0
    in_docstring = False
    
    for i, line in enumerate(lines):
        if line.strip().startswith('"""') or line.strip().startswith("'''"):
            if not in_docstring:
                in_docstring = True
                if line.strip().count('"""') == 2 or line.strip().count("'''") == 2:
                    in_docstring = False
            else:
                in_docstring = False
            continue
            
        if not in_docstring and (line.startswith('import ') or line.startswith('from ')):
            insert_idx = i + 1
            
    if insert_idx == 0:
        for i, line in enumerate(lines):
            if not line.strip().startswith('#') and not line.strip() == '':
                if not line.strip().startswith('"""') and not line.strip().startswith("'''"):
                    insert_idx = i
                    break
                    
    lines.insert(insert_idx, import_statement)
    return '\n'.join(lines)


def refactor():
    files = glob.glob('backend/**/*.py', recursive=True)
    count = 0
    for file in files:
        if 'logger.py' in file.replace('\\', '/'):
            continue
            
        with open(file, 'r', encoding='utf-8') as f:
            content = f.read()
            
        if 'print(' not in content:
            continue
            
        new_content = insert_imports(content)
        
        new_lines = []
        for line in new_content.split('\n'):
            if 'print(' in line and not line.strip().startswith('#'):
                if 'error' in line.lower() or 'fail' in line.lower() or 'exception' in line.lower():
                    line = line.replace('print(', 'logger.error(')
                elif 'warn' in line.lower():
                    line = line.replace('print(', 'logger.warning(')
                else:
                    line = line.replace('print(', 'logger.info(')
            new_lines.append(line)
            
        with open(file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(new_lines))
        count += 1
        print(f'Refactored {file}')
        
    print(f'Total files refactored: {count}')

if __name__ == '__main__':
    refactor()
