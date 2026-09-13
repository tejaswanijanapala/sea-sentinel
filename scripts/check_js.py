import os
import sys

def check_file(path):
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Check for basic JS errors
    stack = []
    pairs = {')': '(', '}': '{', ']': '['}
    in_str = False
    str_char = ''
    in_line_comment = False
    in_block_comment = False
    line = 1
    col = 1
    i = 0
    while i < len(content):
        c = content[i]
        if c == '\n':
            line += 1
            col = 0
            if in_line_comment:
                in_line_comment = False
        col += 1

        if in_line_comment:
            i += 1
            continue
        if in_block_comment:
            if c == '*' and i + 1 < len(content) and content[i+1] == '/':
                in_block_comment = False
                i += 2
                continue
            i += 1
            continue
        if in_str:
            if c == '\\':
                i += 2
                continue
            if c == str_char:
                in_str = False
            i += 1
            continue

        if c == '/' and i + 1 < len(content) and content[i+1] == '/':
            in_line_comment = True
            i += 2
            continue
        if c == '/' and i + 1 < len(content) and content[i+1] == '*':
            in_block_comment = True
            i += 2
            continue
        if c in ('"', "'", '`'):
            in_str = True
            str_char = c
            i += 1
            continue

        if c in '({[':
            stack.append((c, line, col))
        elif c in ')}]':
            if not stack:
                print(f'{path}: Unmatched closing {c} at line {line}:{col}')
                return False
            top, l, cl = stack.pop()
            if pairs[c] != top:
                print(f'{path}: Mismatched {top} from line {l}:{cl} with {c} at line {line}:{col}')
                return False
        i += 1

    if stack:
        print(f'{path}: Unclosed brackets left: {len(stack)}, first unmatched: {stack[0]}')
        return False
    print(f'{path}: All {line} lines syntax/bracket OK!')
    return True

for p in [
    'frontend/js/api.js',
    'frontend/js/waterfall.js',
    'frontend/js/map.js',
    'frontend/shared/js/map.js',
    'frontend/js/gis_ui.js',
    'frontend/js/app.js',
    'frontend/shared/js/app.js'
]:
    if os.path.exists(p):
        check_file(p)
