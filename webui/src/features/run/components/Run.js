import React, { useState } from "react";

import {
  Button,
  Form,
  Input,
  Space,
  Select,
  Modal,
  ConfigProvider
} from "antd";

function Run({run}) {
  const [form] = Form.useForm()

  const onFinish = (values) => {
    // 构造要发送到后端的数据
    const messageToSend = values
    console.log('Request data:', messageToSend);
    // 向后端发送请求...
    fetch('http://localhost:5001/api/configuration/run', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(messageToSend),
    })
      .then(response => {
        if (!response.ok) {
          throw new Error('Network response was not ok');
        }
        return response.json();
      })
      .then(isSucceed => {
        console.log('Message from back-end:', isSucceed);
        Modal.success({
          title: 'Information',
          content: 'Run Successfully!'
        })
      })
      .catch((error) => {
        console.error('Error sending message:', error);
        var errorMessage = error.error;
        Modal.error({
          title: 'Information',
          content: 'Error:' + errorMessage
        })
      });
  };

  return (
    <ConfigProvider
      theme={{
        components: {
          Input: {
            addonBg: "black"
          },
        },
      }}
    >
      <Form
        form={form}
        name="dynamic_form_nest_item"
        onFinish={onFinish}
        style={{ width: "100%" }}
        autoComplete="off"
        initialValues={{
          Seeds: "42",
          Remote: "False"
        }}
      >
        <div style={{ overflowY: 'auto', maxHeight: '150px', display: 'flex', alignItems: 'baseline', justifyContent: 'space-between', gap: 30 }}>
          <div style={{ display: 'flex', alignItems: 'baseline' }}>
            <h6 style={{ color: "black" }}>Seeds</h6>
            <Form.Item name="Seeds" style={{ marginRight: 10, marginLeft: 10 }}>
              <Input />
            </Form.Item>
            <h6 style={{ color: "black" }}>Remote</h6>
            <Form.Item name="Remote" style={{ marginRight: 10, marginLeft: 10 }}>
              <Select
                options={[{ value: "True" },
                { value: "False" },
                ]}
              />
            </Form.Item>
            <h6 style={{ color: "black" }}>ServerURL</h6>
            <Form.Item name="ServerURL" style={{ marginLeft: 10 }}>
              <Input />
            </Form.Item>
          </div>
          <Form.Item>
            <Button 
              type="primary" 
              onClick={() => {
                form.validateFields().then(values => {
                  // 先触发RunPage中的handleRun，然后在onFinish中执行API调用
                  if (typeof run === 'function') {
                    run();
                  }
                  form.submit(); // 提交表单，触发onFinish
                });
              }} 
              style={{ width: "150px", backgroundColor: 'rgb(53, 162, 235)' }}
            >
              Run
            </Button>
          </Form.Item>
        </div>
      </Form>
    </ConfigProvider>
  );
}

export default Run;