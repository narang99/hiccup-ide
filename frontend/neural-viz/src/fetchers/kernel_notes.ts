const apiBaseUrl = "http://localhost:8000";

export interface KernelNoteResponse {
    id: number;
    weight_coordinate: string;
    notes: string;
    created_at: string;
    updated_at: string;
}

export interface KernelNoteRequest {
    notes: string;
}

export async function getKernelNote(weightCoordinate: string): Promise<KernelNoteResponse> {
    const response = await fetch(
        `${apiBaseUrl}/api/kernel-notes/${encodeURIComponent(weightCoordinate)}/`,
        {
            method: 'GET',
        }
    );

    if (!response.ok) {
        throw new Error(`Failed to fetch kernel note: ${response.status}`);
    }

    return response.json();
}

export async function updateKernelNote(
    weightCoordinate: string, 
    notes: string
): Promise<KernelNoteResponse> {
    const response = await fetch(
        `${apiBaseUrl}/api/kernel-notes/${encodeURIComponent(weightCoordinate)}/`,
        {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                notes: notes,
            }),
        }
    );

    if (!response.ok) {
        throw new Error(`Failed to update kernel note: ${response.status}`);
    }

    return response.json();
}

export async function replaceKernelNote(
    weightCoordinate: string, 
    notes: string
): Promise<KernelNoteResponse> {
    const response = await fetch(
        `${apiBaseUrl}/api/kernel-notes/${encodeURIComponent(weightCoordinate)}/`,
        {
            method: 'PUT',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                notes: notes,
            }),
        }
    );

    if (!response.ok) {
        throw new Error(`Failed to replace kernel note: ${response.status}`);
    }

    return response.json();
}