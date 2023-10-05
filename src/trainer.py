import torch   # 適宜importせよ

'''
Day1-2を参考にTrainerクラスを完成させよ
'''
class Trainer():
    def __init__(self, params=None):
        '''
        params: 訓練に関する設定をもつ辞書
        '''
        self.params = params
        self.model = self.get_model(params['model_name'])
        self.optimizer = self.get_optimizer(self.model.parameters(), params['learning_rate'])
        self.criterion = self.get_criterion()            
        
