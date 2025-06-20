
import pytest
import logging
import time
from datetime import datetime
class TestLocalTraining:
    @pytest.mark.dependency(name='local_start')

    def test_train_local_start(self, model_conf, test_session_state, get_model_list, start_training):
        print(f"create_user fixture ID: {id(test_session_state)}")
        # 检查记忆是否上传
        if not test_session_state['memories_uploaded_state']:
            pytest.skip("记忆未上传，跳过训练测试")
        if not test_session_state["uploaded_memories_idList"]:
            pytest.skip("没有找到已上传的记忆，跳过训练测试")
        # 开始云训练
        
        # 1. 获取可训练的模型列表

        modelList_response_body = get_model_list()
        assert modelList_response_body['code'] == 0,"获取模型列表失败"
        assert modelList_response_body['data'], "没有可供训练的模型"

        # 2. 开启训练
        model_id = modelList_response_body["data"][0]['id']

        start_training_url = "http://localhost:3000/api/trainprocess/start"

        payload = {
            "cloud_model_name": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "data_synthesis_mode": "low",
            "concurrency_threads": 2,
            "is_cot": False,
            "language": "chinese",
            "learning_rate": 0.0001,
            "local_model_name": model_id,
            "model_name": model_id,
            "number_of_epochs": 3,
            "use_cuda": False

        }
        start_training_res = start_training(start_training_url,payload)
            
        assert start_training_res['code'] == 0,f'开启训练失败:{start_training_res}' 
        test_session_state["local_training_started"] = True 
                    
    @pytest.mark.dependency(depends=["local_start"])
    def test_train_local_process(self, model_conf, test_session_state, get_model_list, get_training_progress, test_config):
        modelList_response_body = get_model_list()
    
        # 2. 开启训练
        model_id = modelList_response_body["data"][0]['id']
        # 1.获取训练进度
        if not test_session_state.get("local_training_started"):
            pytest.skip("未开启训练")

        # 获取训练进度
        url = f"http://localhost:3000/api/trainprocess/progress/{model_id}"
        start_time = time.time()
        # 2.超出时间限制，发起stop请求
        while True:
            if time.time() - start_time > test_config["timeout_seconds"]:
                logging.info("训练超时，开始尝试停止训练")
                stop_url = "http://localhost:3000/api/trainprocess/stop"
                max_retries = 10
                retry_count = 0
                stop_success = False
            
                while retry_count < max_retries:
                    try:
                        response = requests.post(stop_url)
                        if response.status_code == 200:
                            stop_success = True
                            logging.info("成功发送停止训练请求")
                            assert stop_success == True, "发起停止请求失败"
                            break
                        else:
                            logging.warning(f"停止训练请求失败，状态码: {response.status_code}")
                    except Exception as e:
                        logging.warning(f"停止训练请求异常: {str(e)}")
                
                    retry_count += 1
                    logging.info(f"准备第{retry_count}次重试停止训练请求")
                    time.sleep(1)  # 重试间隔1秒
            
                if not stop_success:
                    pytest.fail(f"尝试{max_retries}次停止训练请求均失败")
                else:
                    pytest.fail("训练超时，已成功发送停止训练请求")
            
            
            
            process_res = get_training_progress(url)
                    

            if process_res['code'] != 0:
                            
                pytest.fail(f"获取训练进度失败：{process_res.get('data','获取不到训练进度的process的data')}")
                    
            process_data = process_res.get("data",{})
            overall_progress = process_data.get("overall_progress",0)
            current_stage = process_data.get("current_stage"," ")
            status = process_data.get("status"," ")
            logging.info(f"训练进度：{overall_progress}%，当前进程为：{current_stage}")

            if overall_progress >= 100:
                logging.info("训练已完成")
                assert status == "completed",f'训练进度100%，训练异常，当前进度{status}'
                break
                    
                    
            time.sleep(test_config["retry_interval"])
            

    

  
