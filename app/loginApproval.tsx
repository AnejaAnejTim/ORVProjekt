import AsyncStorage from '@react-native-async-storage/async-storage';
import React, { useEffect, useState } from 'react';
import {
    ActivityIndicator,
    Alert,
    FlatList,
    StyleSheet,
    Text,
    TouchableOpacity,
    View,
    useColorScheme
} from 'react-native';
import { useRouter } from 'expo-router';

const API_BASE_URL = 'http://192.168.0.13:3001';
const PENDING_LOGINS_URL = `${API_BASE_URL}/login-confirmation/pending`;

interface PendingLogin {
    _id: string;
    confirmationToken: string;
    status: string;
    deviceInfo?: {
        browser?: string;
        os?: string;
    };
}

export default function PendingLoginsScreen() {
    const router = useRouter();
    const isDarkMode = useColorScheme() === 'dark';
    const [pendingLogins, setPendingLogins] = useState<PendingLogin[]>([]);
    const [loading, setLoading] = useState(true);

    const fetchPending = async () => {
        setLoading(true);
        try {
            const jwtToken = await AsyncStorage.getItem('token');
            console.log('fetchPending: token:', jwtToken);
            if (!jwtToken) throw new Error('No JWT token found');
            const res = await fetch(PENDING_LOGINS_URL, {
                method: 'GET',
                headers: {
                    'Authorization': `Bearer ${jwtToken}`,
                    'Content-Type': 'application/json'
                }
            });
            console.log('fetchPending: response status:', res.status);
            const text = await res.text();
            console.log('fetchPending: response text:', text);
            try {
                const data = JSON.parse(text);
                setPendingLogins(data.pendingLogins || []);
            } catch (parseErr) {
                console.log('fetchPending: JSON parse error:', parseErr);
                setPendingLogins([]);
            }
        } catch (err) {
            console.log('fetchPending: error:', err);
            setPendingLogins([]);
        } finally {
            setLoading(false);
        }
    };

    useEffect(() => {
        fetchPending();
    }, []);

    const approveLogin = async (loginId: string) => {
        try {
            const jwtToken = await AsyncStorage.getItem('token');
            if (!jwtToken) throw new Error('No JWT token found');
            
            const res = await fetch(`${API_BASE_URL}/login-confirmation/approve`, {
                method: 'POST',
                headers: {
                    'Authorization': `Bearer ${jwtToken}`,
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ confirmationToken: loginId })
            });
            console.log('approveLogin: response status:', res.status);
            const text = await res.text();
            console.log('approveLogin: response text:', text);
            if (res.ok) {
                Alert.alert('Success', 'Login approved');
                fetchPending();
            } else {
                Alert.alert('Error', 'Failed to approve login');
            }
        } catch (err) {
            console.log('approveLogin: error:', err);
            Alert.alert('Error', 'Failed to approve login');
        }
    };

    const denyLogin = async (loginId: string) => {
        try {
            const jwtToken = await AsyncStorage.getItem('token');
            if (!jwtToken) throw new Error('No JWT token found');
            
            const res = await fetch(`${API_BASE_URL}/login-confirmation/deny`, {
                method: 'POST',
                headers: {
                    'Authorization': `Bearer ${jwtToken}`,
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ confirmationToken: loginId })
            });
            console.log('denyLogin: response status:', res.status);
            const text = await res.text();
            console.log('denyLogin: response text:', text);
            if (res.ok) {
                Alert.alert('Success', 'Login denied');
                fetchPending();
            } else {
                Alert.alert('Error', 'Failed to deny login');
            }
        } catch (err) {
            console.log('denyLogin: error:', err);
            Alert.alert('Error', 'Failed to deny login');
        }
    };

    if (loading) return <ActivityIndicator style={{ flex: 1, justifyContent: 'center' }} />;

    return (
        <View style={[styles.container, isDarkMode && { backgroundColor: '#121212' }]}>
            <TouchableOpacity
                onPress={() => router.back()}
                style={styles.backButton}
            >
                <Text style={styles.backButtonText}>← Nazaj</Text>
            </TouchableOpacity>
            <View style={[styles.card, isDarkMode && { backgroundColor: '#333' }]}>
                <Text style={styles.title}>Čakajoče prijave</Text>
                {pendingLogins.length === 0 ? (
                    <Text style={[styles.emptyText, isDarkMode && { color: '#aaa' }]}>Ni čakajočih prijav.</Text>
                ) : (
                    <FlatList
                        data={pendingLogins}
                        keyExtractor={item => item._id}
                        renderItem={({ item }) => (
                            <View style={[styles.loginItem, isDarkMode && { backgroundColor: '#444' }]}>
                                <Text style={[styles.loginText, isDarkMode && { color: '#eee' }]}>
                                    {item.deviceInfo?.browser || 'Neznana naprava'} - {item.status}
                                </Text>
                                <View style={{ flexDirection: 'row', gap: 12 }}>
                                    <TouchableOpacity
                                        onPress={() => approveLogin(item.confirmationToken)}
                                        style={styles.approveButton}
                                    >
                                        <Text style={styles.approveButtonText}>Potrdi</Text>
                                    </TouchableOpacity>
                                    <TouchableOpacity
                                        onPress={() => denyLogin(item.confirmationToken)}
                                        style={[styles.approveButton, { backgroundColor: '#e57373' }]}
                                    >
                                        <Text style={styles.approveButtonText}>Zavrni</Text>
                                    </TouchableOpacity>
                                </View>
                            </View>
                        )}
                    />
                )}
            </View>
        </View>
    );
}
const styles = StyleSheet.create({
    container: {
        flex: 1,
        backgroundColor: '#f9fafb',
        justifyContent: 'center',
        alignItems: 'center',
        padding: 16,
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
    card: {
        width: '100%',
        maxWidth: 400,
        backgroundColor: '#fff',
        borderRadius: 12,
        padding: 20,
        alignItems: 'center',
        elevation: 4,
    },
    title: {
        fontSize: 20,
        fontWeight: 'bold',
        marginBottom: 16,
        color: '#b0d16b',
    },
    emptyText: {
        color: '#888',
        fontSize: 16,
        marginTop: 24,
    },
    loginItem: {
        width: '100%',
        marginBottom: 16,
        padding: 12,
        borderRadius: 8,
        backgroundColor: '#f0f4e3',
        alignItems: 'center',
    },
    loginText: {
        fontSize: 16,
        marginBottom: 8,
    },
    approveButton: {
        backgroundColor: '#b0d16b',
        paddingVertical: 8,
        paddingHorizontal: 24,
        borderRadius: 6,
        alignItems: 'center',
    },
    approveButtonText: {
        color: 'white',
        fontWeight: 'bold',
        fontSize: 16,
    },
});
