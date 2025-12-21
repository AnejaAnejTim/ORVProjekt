import { Buffer } from 'buffer';
import { CameraView } from 'expo-camera';
import * as FileSystem from 'expo-file-system/legacy';
import { useRouter } from 'expo-router';
import { decode } from 'jpeg-js';
import React, { useRef, useState } from 'react';
import {
  ActivityIndicator,
  Alert,
  Modal,
  StyleSheet,
  Text,
  TextInput,
  TouchableOpacity,
  useColorScheme,
  View,
} from 'react-native';
import { compressImageRGB } from "./Compression";


const Register = (): React.JSX.Element => {
  const isDarkMode = useColorScheme() === 'dark';
  const router = useRouter();

  const [picturesLeft, setPicturesLeft] = useState(0);
  const [cancelCapture, setCancelCapture] = useState(false);

  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [email, setEmail] = useState('');
  const [error, setError] = useState('');

  const cameraRef = useRef<CameraView | null>(null);
  const [cameraVisible, setCameraVisible] = useState(false);
  const [capturing, setCapturing] = useState(false);
  const [faceDataCaptured, setFaceDataCaptured] = useState(false);

  const isValidEmail = (email: string) => {
    const regex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    return regex.test(email);
  };

  const canCaptureFace = email !== '' && username !== '' && password !== '';
  const canRegister = email !== '' && username !== '' && password !== '' && faceDataCaptured;

  const resizeImage = (pixels: Uint8Array, width: number, height: number, newWidth: number, newHeight: number): Uint8Array => {
    const resized = new Uint8Array(newWidth * newHeight * 4);
    const xRatio = width / newWidth;
    const yRatio = height / newHeight;

    for (let y = 0; y < newHeight; y++) {
      for (let x = 0; x < newWidth; x++) {
        const srcX = Math.floor(x * xRatio);
        const srcY = Math.floor(y * yRatio);
        const srcIdx = (srcY * width + srcX) * 4;
        const dstIdx = (y * newWidth + x) * 4;

        resized[dstIdx] = pixels[srcIdx];
        resized[dstIdx + 1] = pixels[srcIdx + 1];
        resized[dstIdx + 2] = pixels[srcIdx + 2];
        resized[dstIdx + 3] = pixels[srcIdx + 3];
      }
    }

    return resized;
  };

  const sendPhotos = async (photos: { base64: string }[], email: string) => {
    const formData = new FormData();

    for (let i = 0; i < photos.length; i++) {
      try {
        console.log(`Processing photo ${i + 1}/${photos.length}...`);
        
        const raw = decode(
          Buffer.from(photos[i].base64, "base64"),
          { useTArray: true }
        );

        const width = raw.width;
        const height = raw.height;
        const pixels = raw.data;
        
        console.log(`  Decoded: ${width}x${height}`);

        // Resize to 300x300
        const targetSize = 300;
        const resizedPixels = resizeImage(pixels, width, height, targetSize, targetSize);
        console.log(`  Resized to: ${targetSize}x${targetSize}`);

        // Separate RGB channels
        const R: number[][] = [];
        const G: number[][] = [];
        const B: number[][] = [];

        let idx = 0;
        for (let x = 0; x < targetSize; x++) {
          R[x] = [];
          G[x] = [];
          B[x] = [];
          for (let y = 0; y < targetSize; y++) {
            R[x][y] = resizedPixels[idx];
            G[x][y] = resizedPixels[idx + 1];
            B[x][y] = resizedPixels[idx + 2];
            idx += 4;
          }
        }

        console.log(`  Separated RGB channels`);

        // Apply custom compression
        const compressed = compressImageRGB(R, G, B);
        console.log(`  Compressed to ${compressed.byteLength} bytes`);

        // Convert compressed ArrayBuffer to base64
        const compressedBuffer = Buffer.from(compressed);
        const compressedBase64 = compressedBuffer.toString('base64');

        // Write to temporary file using legacy API
        const tempFilePath = `${FileSystem.cacheDirectory}face${i}.bin`;
        await FileSystem.writeAsStringAsync(tempFilePath, compressedBase64, {
          encoding: 'base64',
        });
        
        console.log(`  Written to: ${tempFilePath}`);

        // Append file to FormData
        formData.append('images', {
          uri: tempFilePath,
          name: `face${i}.bin`,
          type: 'application/octet-stream',
        } as any);
      } catch (err) {
        console.error(`Error processing photo ${i}:`, err);
        throw new Error(`Failed to process photo ${i}`);
      }
    }

    formData.append('email', email);
    formData.append('username', username);

    console.log('=== Sending to server ===');
    console.log('URL: http://172.20.10.2:5001/');
    
    const res = await fetch('http://172.20.10.2:5001/', {
      method: 'POST',
      body: formData,
    });
    
    console.log(`Response status: ${res.status}`);
    const result = await res.json();
    console.log('Response:', result);

    return result;
  };

  const captureFaceData = async () => {
    try {
      console.log('=== captureFaceData started ===');
      
      if (!canCaptureFace) {
        console.log('Cannot capture face - missing email/username/password');
        Alert.alert('Please fill in Email, Username, and Password first');
        return;
      }

      if (!cameraRef.current) {
        console.log('Camera ref not available');
        return;
      }

      console.log('Starting capture process...');
      setCapturing(true);
      setError('');
      setCancelCapture(false);
      setPicturesLeft(5);

      const photos: { base64: string }[] = [];

      for (let i = 0; i < 5; i++) {
        if (cancelCapture) {
          console.log('Capture cancelled by user');
          setError('Capture cancelled');
          setCapturing(false);
          setCameraVisible(false);
          setPicturesLeft(0);
          return;
        }
        const photo = await cameraRef.current.takePictureAsync({
          quality: 0.8,
          base64: true,
          skipProcessing: true
        });

        if (photo?.base64) {
          photos.push({ base64: photo.base64 });
        }
        
        setPicturesLeft(4 - i);
        
        await new Promise(resolve => setTimeout(resolve, 100));
      }

      
      const result = await sendPhotos(photos, email);


      if (!result.success) {
        console.log('Server returned failure:', result.message);
        setError(result.message || 'Face data capture failed');
        setCapturing(false);
        setPicturesLeft(0);
        return;
      }

      setFaceDataCaptured(true);
      Alert.alert('Face data captured successfully!', 'You can now complete registration.');
    } catch (err) {
      console.error('Error details:', JSON.stringify(err, null, 2));
      setError('Error capturing face data: ' + (err as Error).message);
    } finally {
      setCapturing(false);
      setCameraVisible(false);
      setPicturesLeft(0);
    }
  };

  const handleRegister = async () => {
    if (!faceDataCaptured) {
      Alert.alert('Please capture your face data first.');
      return;
    }

    try {
      const res = await fetch('http://100.117.101.70:3001/users', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          email,
          username,
          password,
        }),
      });

      const data = await res.json();

      if (res.status === 409) {
        setError('Uporabnik že obstaja');
      } else if (res.ok && data._id) {
        Alert.alert('Registracija uspešna', 'Sedaj se lahko prijavite.');
        router.push('/login');
      } else {
        setUsername('');
        setPassword('');
        setEmail('');
        setError(data.message || 'Registracija ni uspela');
      }
    } catch (err) {
      setError('Napaka pri registraciji');
      console.error(err);
    }
  };

  const cancelCaptureProcess = () => {
    if (capturing) {
      setCancelCapture(true);
    }
    setCameraVisible(false);
    setCapturing(false);
    setPicturesLeft(0);
  };

  return (
    <View
      style={[
        styles.container,
        { backgroundColor: isDarkMode ? '#121212' : '#f9fafb' },
      ]}
    >
      <View
        style={[
          styles.card,
          { backgroundColor: isDarkMode ? '#333' : '#fff' },
          isDarkMode ? styles.shadowDark : styles.shadowLight,
        ]}
      >
        <Text style={[styles.title, { color: '#b0d16b' }]}>Registracija</Text>

        <TextInput
          placeholder="Email"
          placeholderTextColor={isDarkMode ? '#aaa' : '#666'}
          style={[
            styles.input,
            {
              color: isDarkMode ? 'white' : 'black',
              borderColor: isDarkMode ? '#555' : '#ccc',
            },
          ]}
          value={email}
          onChangeText={setEmail}
          keyboardType="email-address"
          autoCapitalize="none"
        />

        <TextInput
          placeholder="Uporabniško ime"
          placeholderTextColor={isDarkMode ? '#aaa' : '#666'}
          style={[
            styles.input,
            {
              color: isDarkMode ? 'white' : 'black',
              borderColor: isDarkMode ? '#555' : '#ccc',
            },
          ]}
          value={username}
          onChangeText={setUsername}
          autoCapitalize="none"
        />

        <TextInput
          placeholder="Geslo"
          placeholderTextColor={isDarkMode ? '#aaa' : '#666'}
          style={[
            styles.input,
            {
              color: isDarkMode ? 'white' : 'black',
              borderColor: isDarkMode ? '#555' : '#ccc',
            },
          ]}
          value={password}
          onChangeText={setPassword}
          secureTextEntry
        />

        {error ? <Text style={styles.error}>{error}</Text> : null}

        <TouchableOpacity
          style={[
            styles.button,
            faceDataCaptured ? styles.successButton : {},
            capturing && styles.disabled,
            !canCaptureFace && styles.disabled,
          ]}
          onPress={() => {
            console.log('📷 Opening camera modal');
            setCameraVisible(true);
          }}
          disabled={capturing || !canCaptureFace}
        >
          <Text style={styles.buttonText}>
            {faceDataCaptured ? 'Face Data Captured ✓' : 'Capture Face Data'}
          </Text>
        </TouchableOpacity>

        <TouchableOpacity
          style={[
            styles.button,
            !canRegister && styles.disabled,
          ]}
          onPress={handleRegister}
          disabled={!canRegister}
        >
          <Text style={styles.buttonText}>Registracija</Text>
        </TouchableOpacity>

        <TouchableOpacity onPress={() => router.push('/login')}>
          <Text
            style={[
              styles.loginLink,
              { color: isDarkMode ? '#66ccff' : '#003366' },
            ]}
          >
            Že imate račun? Prijavite se tukaj
          </Text>
        </TouchableOpacity>
      </View>

      <Modal visible={cameraVisible} animationType="slide">
        <View style={{ flex: 1 }}>
          <CameraView style={{ flex: 1 }} facing="front" ref={cameraRef} />

          <View style={styles.ovalOverlay} pointerEvents="none" />

          {capturing && picturesLeft > 0 && (
            <View style={styles.countdownContainer} pointerEvents="none">
              <Text style={styles.countdownText}>Pictures left: {picturesLeft}</Text>
            </View>
          )}

          <View style={{ padding: 20, backgroundColor: '#000' }}>
            {!capturing && (
              <TouchableOpacity 
                style={styles.button} 
                onPress={() => {
                  captureFaceData();
                }}
              >
                <Text style={styles.buttonText}>Start Capturing Photos</Text>
              </TouchableOpacity>
            )}

            <TouchableOpacity
              onPress={cancelCaptureProcess}
              style={[styles.button, { backgroundColor: '#999', marginTop: 10 }]}
            >
              <Text style={styles.buttonText}>
                {capturing ? 'Cancel Capture' : 'Close'}
              </Text>
            </TouchableOpacity>

            {capturing && (
              <ActivityIndicator
                size="large"
                color="#b0d16b"
                style={{ marginTop: 10 }}
              />
            )}
          </View>
        </View>
      </Modal>
    </View>
  );
};

export default Register;

const styles = StyleSheet.create({
  container: {
    flex: 1,
    padding: '5%',
    justifyContent: 'center',
    alignItems: 'center',
  },
  ovalOverlay: {
    position: 'absolute',
    top: '25%',
    left: '15%',
    width: '70%',
    height: '45%',
    borderWidth: 3,
    borderColor: '#b0d16b',
    borderRadius: 1000,
    backgroundColor: 'rgba(176, 209, 107, 0.15)',
  },
  countdownContainer: {
    position: 'absolute',
    top: '45%',
    left: '15%',
    width: '70%',
    height: '10%',
    justifyContent: 'center',
    alignItems: 'center',
    zIndex: 10,
  },
  countdownText: {
    color: '#b0d16b',
    fontSize: 20,
    fontWeight: 'bold',
    backgroundColor: 'rgba(0,0,0,0.5)',
    paddingHorizontal: 15,
    paddingVertical: 8,
    borderRadius: 12,
    textAlign: 'center',
  },
  card: {
    width: '100%',
    maxWidth: 400,
    borderRadius: 12,
    padding: 20,
    alignItems: 'center',
  },
  shadowLight: {
    shadowColor: '#000',
    shadowOpacity: 0.1,
    shadowRadius: 10,
    elevation: 5,
  },
  shadowDark: {
    shadowColor: '#000',
    shadowOpacity: 0.4,
    shadowRadius: 12,
    elevation: 10,
  },
  title: {
    fontSize: 20,
    fontWeight: 'bold',
    marginBottom: 20,
  },
  input: {
    width: '100%',
    borderWidth: 1,
    borderRadius: 8,
    padding: 12,
    marginBottom: 12,
  },
  button: {
    backgroundColor: '#b0d16b',
    padding: 12,
    borderRadius: 8,
    width: '100%',
    alignItems: 'center',
    marginBottom: 12,
  },
  successButton: {
    backgroundColor: '#4caf50',
  },
  disabled: {
    backgroundColor: '#ccc',
  },
  buttonText: {
    color: 'white',
    fontWeight: 'bold',
    fontSize: 16,
  },
  loginLink: {
    fontSize: 14,
    textAlign: 'center',
  },
  error: {
    color: 'red',
    marginBottom: 12,
    textAlign: 'center',
  },
});