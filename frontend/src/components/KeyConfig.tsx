import { useState } from 'react';
import { Box, TextField, Button, Grid, Typography, InputAdornment } from '@mui/material';
import { ConfigData } from '../App';

type KeyConfigProps = {
  config: ConfigData;
  onConfigChange: (config: ConfigData) => void;
};

export default function KeyConfig({ config, onConfigChange }: KeyConfigProps) {
  const [openaiKey, setOpenaiKey] = useState('');
  const [omdbKey, setOmdbKey] = useState('');
  const [modelId, setModelId] = useState(config.model_id || 'liang');
  const [baseUrl, setBaseUrl] = useState(config.base_url || '');
  const [visionModel, setVisionModel] = useState(config.vision_model || '');
  const [visionBaseUrl, setVisionBaseUrl] = useState(config.vision_base_url || '');
  const [status, setStatus] = useState('');

  const handleSave = async () => {
    setStatus('保存中...');
    const payload = {
      openai_api_key: openaiKey || config.openai_api_key,
      model_id: modelId,
      omdb_api_key: omdbKey || config.omdb_api_key,
      base_url: baseUrl,
      vision_model: visionModel,
      vision_base_url: visionBaseUrl,
    };
    const res = await fetch('/api/config', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    const data = await res.json();
    if (data.status === 'ok') {
      setStatus('保存成功');
      onConfigChange({ ...config, ...payload, configured: true });
      setOpenaiKey('');
      setOmdbKey('');
    } else {
      setStatus(`保存失败: ${data.message}`);
    }
  };


  return (
    <Box>
      <Grid container spacing={2} alignItems="end">
        <Grid item xs={12} sm={6}>
          <TextField
            fullWidth
            label="OpenAI API Key"
            value={openaiKey}
            onChange={(e) => setOpenaiKey(e.target.value)}
            placeholder="sk-..."
            type="password"
            InputLabelProps={{ shrink: true }}
            sx={{
              '& .MuiOutlinedInput-root': {
                '& fieldset': { borderColor: 'rgba(15, 23, 42, 0.23)' },
                '&:hover fieldset': { borderColor: '#3b82f6' },
                '&.Mui-focused fieldset': { borderColor: '#3b82f6' },
              },
            }}
          />
        </Grid>
        <Grid item xs={12} sm={6}>
          <TextField
            fullWidth
            label="OMDb API Key"
            value={omdbKey}
            onChange={(e) => setOmdbKey(e.target.value)}
            placeholder="可选"
            type="password"
            InputLabelProps={{ shrink: true }}
            sx={{
              '& .MuiOutlinedInput-root': {
                '& fieldset': { borderColor: 'rgba(15, 23, 42, 0.23)' },
                '&:hover fieldset': { borderColor: '#3b82f6' },
                '&.Mui-focused fieldset': { borderColor: '#3b82f6' },
              },
            }}
          />
        </Grid>
        <Grid item xs={12} sm={6}>
          <TextField
            fullWidth
            label="模型 ID"
            value={modelId}
            onChange={(e) => setModelId(e.target.value)}
            placeholder="gpt-5.4-nano"
            InputLabelProps={{ shrink: true }}
            sx={{
              '& .MuiOutlinedInput-root': {
                '& fieldset': { borderColor: 'rgba(15, 23, 42, 0.23)' },
                '&:hover fieldset': { borderColor: '#3b82f6' },
                '&.Mui-focused fieldset': { borderColor: '#3b82f6' },
              },
            }}
          />
        </Grid>
        <Grid item xs={12} sm={6}>
          <TextField
            fullWidth
            label="API 中转站"
            value={baseUrl}
            onChange={(e) => setBaseUrl(e.target.value)}
            placeholder="https://your-relay.com/v1"
            InputLabelProps={{ shrink: true }}
            sx={{
              '& .MuiOutlinedInput-root': {
                '& fieldset': { borderColor: 'rgba(15, 23, 42, 0.23)' },
                '&:hover fieldset': { borderColor: '#3b82f6' },
                '&.Mui-focused fieldset': { borderColor: '#3b82f6' },
              },
            }}
          />
        </Grid>
        <Grid item xs={12} sm={6}>
          <TextField
            fullWidth
            label="视觉模型（图片描述用）"
            value={visionModel}
            onChange={(e) => setVisionModel(e.target.value)}
            placeholder="如 qwen2.5-vl-7b / gpt-4o-mini"
            helperText="图片分析第一步：把画面转成文字描述，再交给 liang"
            InputLabelProps={{ shrink: true }}
            sx={{
              '& .MuiOutlinedInput-root': {
                '& fieldset': { borderColor: 'rgba(15, 23, 42, 0.23)' },
                '&:hover fieldset': { borderColor: '#3b82f6' },
                '&.Mui-focused fieldset': { borderColor: '#3b82f6' },
              },
            }}
          />
        </Grid>
        <Grid item xs={12} sm={6}>
          <TextField
            fullWidth
            label="视觉模型 API 地址"
            value={visionBaseUrl}
            onChange={(e) => setVisionBaseUrl(e.target.value)}
            placeholder="留空则复用上方中转站"
            InputLabelProps={{ shrink: true }}
            sx={{
              '& .MuiOutlinedInput-root': {
                '& fieldset': { borderColor: 'rgba(15, 23, 42, 0.23)' },
                '&:hover fieldset': { borderColor: '#3b82f6' },
                '&.Mui-focused fieldset': { borderColor: '#3b82f6' },
              },
            }}
          />
        </Grid>
      </Grid>

      <Box sx={{ mt: 2, display: 'flex', alignItems: 'center', gap: 2, flexWrap: 'wrap' }}>
        <Button variant="contained" color="primary" onClick={handleSave}>
          保存配置
        </Button>
        <Typography sx={{ color: '#64748b' }}>{status}</Typography>
      </Box>

      <Box sx={{ mt: 3, p: 3, borderRadius: 2, border: '1px solid rgba(15, 23, 42, 0.08)', bgcolor: '#f9fafb' }}>
        <Typography variant="subtitle2" sx={{ mb: 1, color: '#475569' }}>
          当前配置摘要
        </Typography>
        <Typography variant="body2" sx={{ color: '#111827' }}>
          OpenAI Key: {config.openai_api_key || '未显示或未配置'}
        </Typography>
        <Typography variant="body2" sx={{ color: '#111827' }}>
          OMDb Key: {config.omdb_api_key || '未显示或未配置'}
        </Typography>
        <Typography variant="body2" sx={{ color: '#111827' }}>
          模型 ID: {config.model_id || '未配置'}
        </Typography>
        <Typography variant="body2" sx={{ color: '#111827' }}>
          API 中转站: {config.base_url || '默认'}
        </Typography>
        <Typography variant="body2" sx={{ color: '#111827' }}>
          视觉模型: {config.vision_model || '未配置（图片分析将不可用）'}
        </Typography>
        <Typography variant="body2" sx={{ color: '#111827' }}>
          视觉模型地址: {config.vision_base_url || '复用中转站'}
        </Typography>
      </Box>
    </Box>
  );
}
