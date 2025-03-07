import React, { useState, useEffect } from 'react';
import {
  Container,
  Typography,
  TextField,
  Button,
  Paper,
  Box,
  CircularProgress,
  Alert,
  IconButton,
  ThemeProvider,
  createTheme,
  CssBaseline,
  Fade,
  useMediaQuery,
  Chip,
  Tooltip,
  Divider,
} from '@mui/material';
import { Brightness4, Brightness7, CompareArrows, Info } from '@mui/icons-material';
import axios from 'axios';

interface PredictionResponse {
  similarity_score: number;
  is_duplicate: boolean;
  confidence: string;
  processing_time_ms: number;
}

function App() {
  const [question1, setQuestion1] = useState('');
  const [question2, setQuestion2] = useState('');
  const [result, setResult] = useState<PredictionResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const prefersDarkMode = useMediaQuery('(prefers-color-scheme: dark)');
  const [darkMode, setDarkMode] = useState(prefersDarkMode);

  useEffect(() => {
    setDarkMode(prefersDarkMode);
  }, [prefersDarkMode]);

  const theme = createTheme({
    palette: {
      mode: darkMode ? 'dark' : 'light',
      primary: {
        main: '#2196f3',
      },
      secondary: {
        main: '#f50057',
      },
      background: {
        default: darkMode ? '#121212' : '#f5f5f5',
        paper: darkMode ? '#1e1e1e' : '#ffffff',
      },
    },
    shape: {
      borderRadius: 12,
    },
    components: {
      MuiTextField: {
        defaultProps: {
          variant: 'outlined',
        },
        styleOverrides: {
          root: {
            transition: 'all 0.3s ease-in-out',
          },
        },
      },
      MuiPaper: {
        styleOverrides: {
          root: {
            transition: 'all 0.3s ease-in-out',
          },
        },
      },
      MuiButton: {
        styleOverrides: {
          root: {
            textTransform: 'none',
            transition: 'all 0.3s ease-in-out',
          },
        },
      },
    },
  });

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError('');
    setResult(null);

    try {
      const response = await axios.post<PredictionResponse>('http://localhost:8000/api/predict', {
        question1,
        question2,
      });
      setResult(response.data);
    } catch (err) {
      setError('Error processing request. Please try again.');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  // Function to get color based on similarity score
  const getScoreColor = (score: number) => {
    if (score >= 0.85) return '#4caf50'; // Green
    if (score >= 0.7) return '#8bc34a'; // Light Green
    if (score >= 0.5) return '#ffc107'; // Amber
    if (score >= 0.3) return '#ff9800'; // Orange
    return '#f44336'; // Red
  };

  // Function to get confidence chip color
  const getConfidenceColor = (confidence: string) => {
    switch (confidence) {
      case 'Very High': return '#4caf50';
      case 'High': return '#8bc34a';
      case 'Medium': return '#ffc107';
      case 'Low': return '#ff9800';
      case 'Very Low': return '#f44336';
      default: return '#9e9e9e';
    }
  };

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Container maxWidth="md" sx={{ py: 4, minHeight: '100vh' }}>
        <Box sx={{ display: 'flex', justifyContent: 'flex-end', mb: 2 }}>
          <IconButton onClick={() => setDarkMode(!darkMode)} color="inherit">
            {darkMode ? <Brightness7 /> : <Brightness4 />}
          </IconButton>
        </Box>

        <Fade in={true} timeout={1000}>
          <Box>
            <Typography
              variant="h3"
              component="h1"
              gutterBottom
              align="center"
              sx={{
                fontWeight: 'bold',
                background: darkMode
                  ? 'linear-gradient(45deg, #2196f3, #f50057)'
                  : 'linear-gradient(45deg, #1976d2, #dc004e)',
                backgroundClip: 'text',
                WebkitBackgroundClip: 'text',
                color: 'transparent',
                mb: 2,
              }}
            >
              Quora Query Matcher
            </Typography>
            <Typography
              variant="subtitle1"
              gutterBottom
              align="center"
              color="text.secondary"
              sx={{ mb: 4 }}
            >
              Find semantically similar questions using advanced BERT technology
            </Typography>

            <Paper
              elevation={3}
              sx={{
                p: 4,
                mb: 4,
                borderRadius: 2,
                boxShadow: darkMode
                  ? '0 8px 32px rgba(0, 0, 0, 0.5)'
                  : '0 8px 32px rgba(31, 38, 135, 0.1)',
              }}
            >
              <form onSubmit={handleSubmit}>
                <TextField
                  fullWidth
                  label="Question 1"
                  value={question1}
                  onChange={(e) => setQuestion1(e.target.value)}
                  required
                  sx={{ mb: 3 }}
                  placeholder="Enter the first question"
                />
                <TextField
                  fullWidth
                  label="Question 2"
                  value={question2}
                  onChange={(e) => setQuestion2(e.target.value)}
                  required
                  sx={{ mb: 3 }}
                  placeholder="Enter the second question"
                />
                <Box sx={{ display: 'flex', justifyContent: 'center' }}>
                  <Button
                    type="submit"
                    variant="contained"
                    size="large"
                    disabled={loading}
                    startIcon={<CompareArrows />}
                    sx={{
                      px: 4,
                      py: 1.5,
                      borderRadius: 8,
                      background: 'linear-gradient(45deg, #2196f3, #21cbf3)',
                      '&:hover': {
                        background: 'linear-gradient(45deg, #1976d2, #21cbf3)',
                      },
                    }}
                  >
                    {loading ? 'Processing...' : 'Compare Questions'}
                  </Button>
                </Box>
              </form>
            </Paper>

            {loading && (
              <Box sx={{ display: 'flex', justifyContent: 'center', my: 4 }}>
                <CircularProgress />
              </Box>
            )}

            {error && (
              <Alert severity="error" sx={{ mb: 4 }}>
                {error}
              </Alert>
            )}

            {result && (
              <Fade in={true} timeout={1000}>
                <Paper
                  elevation={3}
                  sx={{
                    p: 4,
                    borderRadius: 2,
                    boxShadow: darkMode
                      ? '0 8px 32px rgba(0, 0, 0, 0.5)'
                      : '0 8px 32px rgba(31, 38, 135, 0.1)',
                  }}
                >
                  <Typography variant="h5" gutterBottom align="center" sx={{ mb: 3 }}>
                    Similarity Results
                  </Typography>

                  <Box sx={{ display: 'flex', justifyContent: 'center', mb: 3 }}>
                    <Box
                      sx={{
                        position: 'relative',
                        display: 'inline-flex',
                        justifyContent: 'center',
                        alignItems: 'center',
                      }}
                    >
                      <CircularProgress
                        variant="determinate"
                        value={result.similarity_score * 100}
                        size={120}
                        thickness={5}
                        sx={{
                          color: getScoreColor(result.similarity_score),
                          '& .MuiCircularProgress-circle': {
                            strokeLinecap: 'round',
                          },
                        }}
                      />
                      <Box
                        sx={{
                          top: 0,
                          left: 0,
                          bottom: 0,
                          right: 0,
                          position: 'absolute',
                          display: 'flex',
                          alignItems: 'center',
                          justifyContent: 'center',
                        }}
                      >
                        <Typography
                          variant="h4"
                          component="div"
                          sx={{ fontWeight: 'bold' }}
                        >
                          {Math.round(result.similarity_score * 100)}%
                        </Typography>
                      </Box>
                    </Box>
                  </Box>

                  <Box sx={{ display: 'flex', justifyContent: 'center', mb: 3, gap: 2, flexWrap: 'wrap' }}>
                    <Chip
                      label={result.is_duplicate ? 'Duplicate' : 'Not Duplicate'}
                      color={result.is_duplicate ? 'success' : 'error'}
                      variant="outlined"
                    />
                    <Chip
                      label={`Confidence: ${result.confidence}`}
                      sx={{ 
                        backgroundColor: getConfidenceColor(result.confidence),
                        color: 'white'
                      }}
                    />
                    <Tooltip title="Time taken to process the request">
                      <Chip
                        icon={<Info />}
                        label={`${result.processing_time_ms.toFixed(2)} ms`}
                        variant="outlined"
                      />
                    </Tooltip>
                  </Box>

                  <Divider sx={{ my: 2 }} />

                  <Typography variant="body1" align="center" sx={{ mt: 2 }}>
                    {result.is_duplicate
                      ? 'These questions are semantically similar and likely duplicates.'
                      : 'These questions are semantically different and not duplicates.'}
                  </Typography>
                </Paper>
              </Fade>
            )}
          </Box>
        </Fade>
      </Container>
    </ThemeProvider>
  );
}

export default App;
