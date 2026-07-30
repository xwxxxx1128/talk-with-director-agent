import { useEffect, useState, type Dispatch, type SetStateAction } from 'react';
import { Box, Container, CssBaseline, Typography, Paper, Divider, Button } from '@mui/material';
import Header from './components/Header';
import KeyConfig from './components/KeyConfig';
import MessagePanel from './components/MessagePanel';

export type ConfigData = {
  openai_api_key: string | null;
  model_id: string | null;
  omdb_api_key: string | null;
  base_url: string | null;
  vision_model: string | null;
  vision_base_url: string | null;
  configured: boolean;
};

export type Message = {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: number;
  image?: string;
};

export type MessageSetter = Dispatch<SetStateAction<Message[]>>;

const defaultConfig: ConfigData = {
  openai_api_key: null,
  model_id: 'liang',
  omdb_api_key: null,
  base_url: null,
  vision_model: null,
  vision_base_url: null,
  configured: false,
};

function App() {
  const [config, setConfig] = useState<ConfigData>(defaultConfig);
  const [messages, setMessages] = useState<Message[]>([]);
  const [configOpen, setConfigOpen] = useState(false);

  useEffect(() => {
    fetch('/api/config')
      .then((res) => res.json())
      .then((data) => setConfig(data))
      .catch(console.error);
  }, []);

  useEffect(() => {
    if (!config.configured) {
      setConfigOpen(true);
    }
  }, [config.configured]);

  return (
    <Box sx={{ minHeight: '100vh', bgcolor: '#f9fafb', color: '#111827' }}>
      <CssBaseline />
      <Header configured={config.configured} />
      <Container maxWidth="lg" sx={{ py: 4 }}>
        {config.configured && !configOpen ? (
          <Paper sx={{ p: 3, mb: 4, bgcolor: '#ffffff', border: '1px solid rgba(15, 23, 42, 0.08)', boxShadow: 'none' }}>
            <Typography variant="h6" gutterBottom>
              当前已配置，想要重新设置？
            </Typography>
            <Button variant="contained" color="primary" onClick={() => setConfigOpen(true)}>
              重新配置
            </Button>
          </Paper>
        ) : (
          <Paper sx={{ p: 3, mb: 4, bgcolor: '#ffffff', border: '1px solid rgba(15, 23, 42, 0.08)', boxShadow: 'none' }}>
            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2, flexWrap: 'wrap', gap: 2 }}>
              <Box>
                <Typography variant="h5" gutterBottom>
                  系统配置面板
                </Typography>
                <Typography variant="body2" sx={{ color: '#475569' }}>
                  当前已加载的 API Key 和模型信息，可在企业控制台风格界面中方便地查看与更新。
                </Typography>
              </Box>
              {config.configured && (
                <Button variant="outlined" color="primary" onClick={() => setConfigOpen(false)}>
                  收起配置
                </Button>
              )}
            </Box>
            <KeyConfig config={config} onConfigChange={setConfig} />
          </Paper>
        )}

        <Paper sx={{ p: 3, bgcolor: '#ffffff', boxShadow: '0 18px 38px rgba(15,23,42,0.08)' }}>
          <Typography variant="h5" gutterBottom>
            智能对话与镜头分析
          </Typography>
          <MessagePanel
            config={config}
            messages={messages}
            setMessages={setMessages}
          />
        </Paper>
      </Container>
    </Box>
  );
}

export default App;
