import AsyncStorage from '@react-native-async-storage/async-storage';
import {useNavigation} from '@react-navigation/native';
import {CameraView, useCameraPermissions} from 'expo-camera';
import * as ImageManipulator from 'expo-image-manipulator';
import {useRouter} from 'expo-router';
import React, {useEffect, useRef, useState} from 'react';
import {ActivityIndicator, Alert, StyleSheet, Text, TouchableOpacity, View} from 'react-native';
export default function FaceAuth() {
    const cameraRef = useRef<CameraView>(null);
    const router = useRouter();
    const [permission, requestPermission] = useCameraPermissions();
    const [loading, setLoading] = useState(false);
    const navigation = useNavigation();

    useEffect(() => {
        if (!permission || !permission.granted) {
            requestPermission();
        }
    }, [permission]);

    const handleScan = async () => {
        if (!cameraRef.current) return;

        setLoading(true);

        try {
            const photo = await cameraRef.current.takePictureAsync({
                quality: 1,
                base64: false,
            });

            const resizedPhoto = await ImageManipulator.manipulateAsync(
                photo.uri,
                [{resize: {width: 300, height: 300}}],
                {compress: 0.8, format: ImageManipulator.SaveFormat.JPEG}
            );

            const token = await AsyncStorage.getItem('token');
            const email = await AsyncStorage.getItem('email');

            if (!token) {
                Alert.alert('Error', 'Authentication token not found');
                setLoading(false);
                return;
            }

            if (!email) {
                Alert.alert('Error', 'Email not found in local storage');
                setLoading(false);
                return;
            }

            const formData = new FormData();
            formData.append('email', email);
            formData.append('image', {
                uri: resizedPhoto.uri,
                type: 'image/jpeg',
                name: 'face.jpg',
            } as any);

            const res = await fetch('http://localhost:5001/authenticateFace', {
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

    if (!permission) return <View style={styles.container}><ActivityIndicator /></View>;
    if (!permission.granted) return <View style={styles.container}><Text>No access to camera</Text></View>;

    return (
        <View style={{flex: 1}}>
            <TouchableOpacity
                onPress={() => navigation.goBack()}
                style={styles.backButton}
            >
                <Text style={styles.backButtonText}>← Back</Text>
            </TouchableOpacity>

            <CameraView style={{flex: 1}} ref={cameraRef} facing="front"/>
            <TouchableOpacity style={styles.button} onPress={handleScan} disabled={loading}>
                {loading ? <ActivityIndicator color="#fff"/> : <Text style={styles.buttonText}>Scan Face</Text>}
            </TouchableOpacity>
        </View>
    );
}

const styles = StyleSheet.create({
    container: {
        flex: 1,
        justifyContent: 'center',
        alignItems: 'center',
        backgroundColor: '#000',
    },
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
    buttonText: {color: '#fff', fontSize: 18, fontWeight: 'bold'}
});