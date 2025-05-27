import AsyncStorage from '@react-native-async-storage/async-storage';
import { useNavigation } from '@react-navigation/native';
import { Camera, CameraView } from 'expo-camera';
import { useRouter } from 'expo-router';
import React, { useEffect, useRef, useState } from 'react';
import { ActivityIndicator, Alert, StyleSheet, Text, TouchableOpacity, View } from 'react-native';

export default function FaceAuth() {
  const cameraRef = useRef(null);
  const router = useRouter();
  const [hasPermission, setHasPermission] = useState(null);
  const [loading, setLoading] = useState(false);
    const navigation = useNavigation();

  useEffect(() => {
    (async () => {
      const { status } = await Camera.requestCameraPermissionsAsync();
      setHasPermission(status === 'granted');
    })();
  }, []);

  const handleScan = async () => {
  if (!cameraRef.current) return;

  setLoading(true);

  try {
    const photo = await cameraRef.current.takePictureAsync({
      quality: 0.3,
      base64: false,
    });

    const token = await AsyncStorage.getItem('token');
    const email = await AsyncStorage.getItem('email');

    if (!email) {
      Alert.alert('Error', 'Email not found in local storage');
      setLoading(false);
      return;
    }

    const formData = new FormData();
    formData.append('email', email);
    formData.append('image', {
      uri: photo.uri,
      type: 'image/jpeg',
      name: 'face.jpg',
    });

    const res = await fetch('http://100.117.101.70:5001/authenticateFace', {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${token}`,
      },
      body: formData,
    });

    const result = await res.json();

    if (res.ok && result.success) {
      router.replace('/loginApproval');
    } else {
      Alert.alert('Authentication Failed', result.message || 'Face not recognized');
    }

  } catch (err) {
    console.log('faceAuth error:', err);
    Alert.alert('Error', 'Could not verify your face');
  } finally {
    setLoading(false);
  }
};



  if (hasPermission === null) return <View><Text>Requesting camera access...</Text></View>;
  if (hasPermission === false) return <View><Text>No access to camera</Text></View>;

  return (
    <View style={{ flex: 1 }}>
         <TouchableOpacity
              onPress={() => navigation.goBack()}
              style={styles.backButton}
            >
              <Text style={styles.backButtonText}>← Back</Text>
            </TouchableOpacity>
        
      <CameraView style={{ flex: 1 }} ref={cameraRef} facing="front"/>
      <TouchableOpacity style={styles.button} onPress={handleScan} disabled={loading}>
        {loading ? <ActivityIndicator color="#fff" /> : <Text style={styles.buttonText}>Scan Face</Text>}
      </TouchableOpacity>
    </View>
  );
}

const styles = StyleSheet.create({
  button: {
    position: 'absolute',
    bottom: 40,
    alignSelf: 'center',
    backgroundColor: '#4caf50',
    paddingVertical: 12,
    paddingHorizontal: 32,
    borderRadius: 30,
  },
    backButton: {
    position: 'absolute',
    top: 50,
    left: 12,
    paddingVertical: 6,
    paddingHorizontal: 12,
    zIndex: 10,
  },
  backButtonText: {
    fontSize: 16,
    color: '#66ccff',
    fontWeight: 'bold',
  },
  buttonText: { color: '#fff', fontSize: 18, fontWeight: 'bold' }
});