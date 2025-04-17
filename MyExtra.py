VALID_POS = [
    (2, 2), (3, 2), (4, 2), (5, 2), (6, 2),
    (2, 3), (4, 3), (6, 3),
    (2, 4), (3, 4), (4, 4), (5, 4), (6, 4),
    (2, 5), (4, 5), (6, 5),
    (2, 6), (3, 6), (4, 6), (5, 6), (6, 6), (1, 4), (7, 4) ]

def checkpos(obs):
    try:
        #entities = obs[1]
        state = obs[0]
        print('state:', state)

        me = [(j, i) for i, v in enumerate(state) for j, k in enumerate(v) if 'Agent_2' in k]
        you = [(j, i) for i, v in enumerate(state) for j, k in enumerate(v) if 'Agent_1' in k]
        target = [(j, i) for i, v in enumerate(state) for j, k in enumerate(v) if 'Pig' in k]
        print('me:', me, 'you:', you, 'target:', target)
        # 提取坐标部分进行比较
        me_pos = me[0][0:2] if me else None
        you_pos = you[0][0:2] if you else None
        target_pos = target[0][0:2] if target else None

        if (me_pos not in VALID_POS) or (you_pos not in VALID_POS) or (target_pos not in VALID_POS):
            print('位置不对')
            return False
        else: 
            print('位置对')
            return True
        
    except:

        print('检测错误   Error in checkpos')
        return False
