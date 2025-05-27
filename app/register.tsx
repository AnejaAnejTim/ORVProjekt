import { CameraView } from 'expo-camera';
import * as ImageManipulator from 'expo-image-manipulator';
import { useRouter } from 'expo-router';
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

  const sendPhotos = async (photoUris: string[], email: string) => {
    const formData = new FormData();

    for (let i = 0; i < photoUris.length; i++) {
      const uri = photoUris[i];

      const manipulated = await ImageManipulator.manipulateAsync(
        uri,
        [{ resize: { width: 300, height: 300 } }],
        { compress: 0.8, format: ImageManipulator.SaveFormat.JPEG }
      );

      formData.append('images', {
        uri: manipulated.uri,
        type: 'image/jpeg',
        name: `face${i}.jpg`,
      } as any);
    }

    formData.append('email', email);
    formData.append('username', username);

    const res = await fetch('http://100.117.101.70:5001/register-face', {
      method: 'POST',
      headers: {
        'Content-Type': 'multipart/form-data',
      },
      body: formData,
    });

    return res.json();
  };

  const captureFaceData = async () => {
    if (!canCaptureFace) {
      Alert.alert('Please fill in Email, Username, and Password first');
      return;
    }

    if (!cameraRef.current) return;

    setCapturing(true);
    setError('');
    setCancelCapture(false);
    setPicturesLeft(40);

    try {
      const photoUris: string[] = [];

      for (let i = 0; i < 40; i++) {
        if (cancelCapture) {
          setError('Capture cancelled');
          setCapturing(false);
          setCameraVisible(false);
          setPicturesLeft(0);
          return;
        }

        const photo = await cameraRef.current.takePictureAsync({
          quality: 0.3,
          base64: false,
        });

        photoUris.push(photo.uri);
        setPicturesLeft(39 - i);
      }

      const result = await sendPhotos(photoUris, email);

      if (!result.success) {
        setError(result.message || 'Face data capture failed');
        setCapturing(false);
        setPicturesLeft(0);
        return;
      }

      setFaceDataCaptured(true);
      Alert.alert('Face data captured successfully!', 'You can now complete registration.');
    } catch (err) {
      setError('Error capturing face data');
      console.error(err);
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

    if (!isValidEmail(email)) {
      setError('Prosim vnesite veljaven email naslov.');
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
          onPress={() => setCameraVisible(true)}
          disabled={capturing || !canCaptureFace}
        >
          <Text style={styles.buttonText}>
            {faceDataCaptured ? 'Face Data Captured ✓' : 'Capture Face Data'}
          </Text>
        </TouchableOpacity>

        <TouchableOpacity
          style={[
            styles.button,
            !(email && username && password && faceDataCaptured) && styles.disabled,
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

          {/* Countdown centered inside the oval */}
          {capturing && picturesLeft > 0 && (
            <View style={styles.countdownContainer} pointerEvents="none">
              <Text style={styles.countdownText}>Pictures left: {picturesLeft}</Text>
            </View>
          )}

          <View style={{ padding: 20, backgroundColor: '#000' }}>
            {!capturing && (
              <TouchableOpacity style={styles.button} onPress={captureFaceData}>
                <Text style={styles.buttonText}>Start Capturing 40 Photos</Text>
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
    top: '45%', // roughly vertical center of the oval
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
