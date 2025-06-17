import pytest
import requests
import os
from pathlib import Path
import json
import time
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
# 定义测试参数，包括单个文件和文件夹路径

class TestTrain:
    start_train_states = False
    def get_modelList(self):
        try:
            # 1. 获取可训练的模型列表
            get_modelId_url = "http://localhost:3000/api/cloud_service/list_available_models"
            response = requests.get(get_modelId_url)
            response.raise_for_status()
            modelList_response_body = response.json()
            return modelList_response_body
        except Exception as e:
            print(f"执行错误: {e}")
            raise
    

    @pytest.mark.dependency(name='test_train_cloud_start')
    def test_train_cloud_start(self,model_conf,test_session_state):
        # 检查记忆是否上传
        if not test_session_state['memories_uploaded_state']:
            pytest.skip("记忆未上传，跳过训练测试")
        if not test_session_state["uploaded_memories_idList"]:
            pytest.skip("没有找到已上传的记忆，跳过训练测试")
        # 开始云训练
        try:
            # 1. 获取可训练的模型列表

            modelList_response_body = self.get_modelList()
            assert modelList_response_body['code'] == 0,"获取模型列表失败"
            assert len(modelList_response_body['data'])>0,"没有可供训练的模型"

            start_training_url = "http://localhost:3000/api/cloud_service/train/start"
            headers = {
   'Content-Type': 'application/json',
   'Accept': '*/*',
   'Connection': 'keep-alive'
}
            payload = json.dumps({
    
   "base_model": modelList_response_body["data"][0]['id'],
   "data_synthesis_mode": "low",
   "hyper_parameters": {
       "n_epochs": 3,
       "learning_rate": 0.0001
   },
   "training_type": "efficient_sft"
})
            response = requests.post(start_training_url,headers=headers, data=payload)
            response.raise_for_status()
            start_training_res = response.json()
            
            assert start_training_res['code'] == 0,f'开启训练失败:{start_training_res}' 
            TestTrain.start_train_states = True
        except Exception as e:
            print(f"执行错误: {e}")
            TestTrain.start_train_states = False
            raise


    @pytest.mark.dependency(depends=["test_train_cloud_start"],scope = "class")
    def test_train_cloud_nonStop(self,model_conf,test_session_state):
        """"测试1.不中断训练"""
        if TestTrain.start_train_states:
            # 2. 轮询训练进度，直到完成或出错或超出时间
            try:
                process_url = "http://localhost:3000/api/cloud_service/train/progress"
                stop_url = "http://localhost:3000/api/cloud_service/train/stop"

                retry_interval = 5
                timeout_seconds = 7200
                stop_time = 120
                
                start_time = time.time()
                while True:
                    if time.time() - start_time > timeout_seconds:
                        pytest.fail("训练超时")
                
                    try:
                        response = requests.get(process_url)
                        response.raise_for_status()
                        process_res = response.json()

                        if process_res['code'] != 0:
                            pytest.fail(f"获取训练进度失败：{process_res.get("data","获取不到训练进度的process的data")}")
                    
                        process_data = process_res.get("data","获取不到训练进度的process的data").get("progress"," ")
                        overall_progress = process_data.get("overall_progress"," ")
                        current_stage = process_data.get("current_stage","")
                        print(f"训练进度：{overall_progress}%，当前进程为：{current_stage}")

                        if overall_progress >= 100.0:
                            print("训练已完成")
                            return process_res
                        time.sleep(retry_interval)
                    except requests.RequestException as e:
                        print(f"请求训练进度失败: {e}")
                        time.sleep(retry_interval)
                    except Exception as e:
                        print(f"处理训练进度响应失败: {e}")
                        time.sleep(retry_interval)       
            except Exception as e:
                print(f"执行错误: {e}")
                raise
        else:
            pytest.skip("未开启训练")

    

    # def test_train_cloud_stop(self,model_conf,test_session_state):
    #     """"测试2.微调前中断训练"""
    #     if TestTrain.start_train_states:
    #         # 2. 轮询训练进度，直到完成或出错或超出时间

    #         try:
    #             process_url = "http://localhost:3000/api/cloud_service/train/progress"
    #             retry_interval = 1
    #             timeout_seconds = 7200
    #             start_time = time.time()
    #             while True:
    #                 if time.time() - start_time > timeout_seconds:
    #                     pytest.fail("训练超时")
                
    #                 try:
    #                     response = requests.get(process_url)
    #                     response.raise_for_status()
    #                     process_res = response.json()

    #                     if process_res['code'] != 0:
    #                         pytest.fail(f"获取训练进度失败：{process_res.get("data","获取不到训练进度的process的data")}")
                    
    #                     process_data = process_res.get("data","获取不到训练进度的process的data").get("progress"," ")
    #                     overall_progress = process_data.get("overall_progress"," ")
    #                     current_stage = process_data.get("current_stage","")
    #                     print(f"训练进度：{overall_progress}%，当前进程为：{current_stage}")

    #                     if overall_progress >= 100.0:
    #                         print("训练已完成")
    #                         return process_res
    #                     time.sleep(retry_interval)
    #                 except requests.RequestException as e:
    #                     print(f"请求训练进度失败: {e}")
    #                     time.sleep(retry_interval)
    #                 except Exception as e:
    #                     print(f"处理训练进度响应失败: {e}")
    #                     time.sleep(retry_interval)       
    #         except Exception as e:
    #             print(f"执行错误: {e}")
    #             raise
    #     else:
    #         pytest.skip("未开启训练")